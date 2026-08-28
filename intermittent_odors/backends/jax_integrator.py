import os

import jax
import jax.numpy as jnp

from intermittent_odors.backends.jax_precision import JAX_DTYPE

#: Unroll factor for the time-stepping scans. Unrolling amortizes loop overhead
#: without reordering any floating-point operation, so results stay bit-for-bit
#: identical to unroll=1; the cost is compile time (~4x at 8), which the
#: persistent compilation cache pays once. Measured gains in docs/performance.md.
SCAN_UNROLL = max(1, int(os.environ.get('IODOR_JAX_SCAN_UNROLL', '8')))

#: Opt-in sampling strategy for integrate_sampled: a nested scan (inner scan of
#: sample_stride steps, outer scan emitting one sample per chunk) instead of the
#: per-step cond + dynamic_update_slice write. Faster, especially on GPU, but it
#: changes XLA fusion and drifts from the default path by ~1e-22 — NOT bitwise
#: identical, so it must never be used to regenerate committed datasets.
NESTED_SAMPLING = os.environ.get('IODOR_JAX_NESTED_SAMPLING', '').strip().lower() in {'1', 'true', 'yes'}


def jax_check_type(t, y0):
    if not (jnp.issubdtype(y0.dtype, jnp.floating) and jnp.issubdtype(t.dtype, jnp.floating)):
        raise TypeError('Error in Datatype')


class _JaxIntegrator():

    def __init__(self, n_, F_b):
        self.n_ = n_
        self.F_b = F_b

    def _advance(self, func, t, dt, y):
        """One RK4 step plus the spike-time update. Every integration path must
        go through here so that all of them perform the identical sequence of
        floating-point operations per step."""
        dy = self._step_func(func, t, dt, y)
        dy = jnp.asarray(dy, dtype=y.dtype)

        out = y + dy

        n_ = self.n_
        F_b = self.F_b
        if n_ > 0:
            fire_t = y[-n_:]
            l = jnp.zeros_like(fire_t)
            l_ = t - fire_t
            z = y[:n_] < F_b
            z_ = out[:n_] >= F_b
            df = jnp.where(jnp.logical_and(z, z_), l_, l)
            fire_t_ = fire_t + df
            out = jnp.concatenate([out[:-n_], fire_t_], axis=0)

        return out

    def integrate(self, func, y0, t):
        time_delta_grid = t[1:] - t[:-1]

        def scan_func(y, t_dt):
            t, dt = t_dt
            out = self._advance(func, t, dt, y)
            return out, out

        _, y = jax.lax.scan(scan_func, y0, (t[:-1], time_delta_grid), unroll=SCAN_UNROLL)
        return jnp.concatenate([y0[None, :], y], axis=0)

    def integrate_sampled(self, func, y0, t, sample_stride, sample_neurons, sample_phase=0):
        total_points = t.shape[0]
        if total_points <= 1:
            return y0[None, :sample_neurons], y0

        total_steps = total_points - 1
        # Samples land on local step indices sample_phase, sample_phase + stride, ...
        # A non-zero phase means this batch starts part-way through a sampling
        # interval, which is what keeps the grid uniform when a long rollout is
        # split into overlapping time batches. The batch's final point is always
        # left to the next batch, which begins on it.
        sample_phase = int(sample_phase) % int(sample_stride)
        sample_count = max(0, -(-(total_steps - sample_phase) // sample_stride))

        if NESTED_SAMPLING:
            return self._integrate_sampled_nested(
                func, y0, t, sample_stride, sample_neurons, sample_phase, sample_count
            )

        # A batch shorter than the gap to the next grid point contributes no
        # rows. Keep a one-row scratch buffer so the scan body still traces, and
        # slice it away on the way out.
        buffer_rows = max(sample_count, 1)
        time_delta_grid = t[1:] - t[:-1]
        samples = jnp.zeros((buffer_rows, sample_neurons), dtype=y0.dtype)
        if sample_phase == 0:
            samples = samples.at[0].set(y0[:sample_neurons])
            first_slot = 1
        else:
            first_slot = 0

        def scan_func(carry, t_dt):
            y, sampled, sample_slot, step_index = carry
            t, dt = t_dt
            out = self._advance(func, t, dt, y)

            should_sample = jnp.logical_and(
                jnp.equal((step_index + 1) % sample_stride, sample_phase),
                step_index + 1 < total_steps,
            )

            def write_sample(buffer):
                return jax.lax.dynamic_update_slice(
                    buffer,
                    out[:sample_neurons][None, :],
                    (sample_slot, jnp.asarray(0, dtype=jnp.int32)),
                )

            sampled = jax.lax.cond(should_sample, write_sample, lambda buffer: buffer, sampled)
            sample_slot = sample_slot + should_sample.astype(jnp.int32)
            return (out, sampled, sample_slot, step_index + 1), None

        carry = (y0, samples, jnp.asarray(first_slot, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        (final_state, sampled, _, _), _ = jax.lax.scan(
            scan_func, carry, (t[:-1], time_delta_grid), unroll=SCAN_UNROLL
        )
        return sampled[:sample_count], final_state

    def _integrate_sampled_nested(self, func, y0, t, sample_stride, sample_neurons, sample_phase, sample_count):
        """Sampling via nested scans instead of a per-step cond + buffer write.

        The rollout is split into a head of ``sample_phase`` steps, then
        ``sample_count - 1`` chunks of ``sample_stride`` steps whose end states
        the outer scan emits, then a sample-free tail. Sample positions are
        identical to the cond path, and every step goes through the same
        ``_advance``, but the different scan structure changes XLA fusion, so
        the emitted values can differ from the cond path in the last bits.
        """
        total_steps = t.shape[0] - 1
        time_delta_grid = t[1:] - t[:-1]

        def one_step(y, t_dt):
            t, dt = t_dt
            return self._advance(func, t, dt, y), None

        def run_steps(y, start, stop):
            if stop <= start:
                return y
            y, _ = jax.lax.scan(
                one_step, y, (t[start:stop], time_delta_grid[start:stop]), unroll=SCAN_UNROLL
            )
            return y

        if sample_count == 0:
            final_state = run_steps(y0, 0, total_steps)
            return jnp.zeros((0, sample_neurons), dtype=y0.dtype), final_state

        if sample_phase == 0:
            state = y0
        else:
            state = run_steps(y0, 0, sample_phase)
        first_row = state[:sample_neurons]

        n_chunks = sample_count - 1
        chunk_stop = sample_phase + n_chunks * sample_stride

        def emit_chunk(y, chunk):
            y, _ = jax.lax.scan(one_step, y, chunk, unroll=SCAN_UNROLL)
            return y, y[:sample_neurons]

        chunk_ts = t[sample_phase:chunk_stop].reshape(n_chunks, sample_stride)
        chunk_dts = time_delta_grid[sample_phase:chunk_stop].reshape(n_chunks, sample_stride)
        state, emits = jax.lax.scan(emit_chunk, state, (chunk_ts, chunk_dts))

        final_state = run_steps(state, chunk_stop, total_steps)
        sampled = jnp.concatenate([first_row[None, :], emits], axis=0)
        return sampled, final_state

    def _step_func(self, func, t, dt, y):
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = jnp.asarray(dt, dtype=y.dtype)

        k2 = func(y + dt_cast * k1 / 2, half_step)
        k3 = func(y + dt_cast * k2 / 2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        return (k1 + 2 * k2 + 2 * k3 + k4) * (dt_cast / 6)


def odeint(func, y0, t, n_, F_b):
    t = jnp.asarray(t, dtype=JAX_DTYPE)
    y0 = jnp.asarray(y0, dtype=JAX_DTYPE)
    jax_check_type(t, y0)
    F_b = jnp.asarray(F_b, dtype=y0.dtype)
    return _JaxIntegrator(n_, F_b).integrate(func, y0, t)


def odeint_sampled(func, y0, t, n_, F_b, sample_stride, sample_neurons, sample_phase=0):
    t = jnp.asarray(t, dtype=JAX_DTYPE)
    y0 = jnp.asarray(y0, dtype=JAX_DTYPE)
    jax_check_type(t, y0)
    F_b = jnp.asarray(F_b, dtype=y0.dtype)
    return _JaxIntegrator(n_, F_b).integrate_sampled(
        func, y0, t, sample_stride, sample_neurons, sample_phase
    )