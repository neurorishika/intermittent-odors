import jax
import jax.numpy as jnp

from intermittent_odors.jax_precision import JAX_DTYPE


def jax_check_type(t, y0):
    if not (jnp.issubdtype(y0.dtype, jnp.floating) and jnp.issubdtype(t.dtype, jnp.floating)):
        raise TypeError('Error in Datatype')


class _JaxIntegrator():

    def __init__(self, n_, F_b):
        self.n_ = n_
        self.F_b = F_b

    def integrate(self, func, y0, t):
        time_delta_grid = t[1:] - t[:-1]

        def scan_func(y, t_dt):
            n_ = self.n_
            F_b = self.F_b

            t, dt = t_dt

            dy = self._step_func(func, t, dt, y)
            dy = jnp.asarray(dy, dtype=y.dtype)

            out = y + dy

            if n_ > 0:
                fire_t = y[-n_:]
                l = jnp.zeros_like(fire_t)
                l_ = t - fire_t
                z = y[:n_] < F_b
                z_ = out[:n_] >= F_b
                df = jnp.where(jnp.logical_and(z, z_), l_, l)
                fire_t_ = fire_t + df
                out = jnp.concatenate([out[:-n_], fire_t_], axis=0)

            return out, out

        _, y = jax.lax.scan(scan_func, y0, (t[:-1], time_delta_grid))
        return jnp.concatenate([y0[None, :], y], axis=0)

    def integrate_sampled(self, func, y0, t, sample_stride, sample_neurons):
        total_points = t.shape[0]
        if total_points <= 1:
            return y0[None, :sample_neurons], y0

        total_steps = total_points - 1
        sample_count = 1 + ((total_points - 2) // sample_stride)
        time_delta_grid = t[1:] - t[:-1]
        samples = jnp.zeros((sample_count, sample_neurons), dtype=y0.dtype)
        samples = samples.at[0].set(y0[:sample_neurons])

        def scan_func(carry, t_dt):
            y, sampled, sample_slot, step_index = carry
            n_ = self.n_
            F_b = self.F_b

            t, dt = t_dt
            dy = self._step_func(func, t, dt, y)
            dy = jnp.asarray(dy, dtype=y.dtype)

            out = y + dy

            if n_ > 0:
                fire_t = y[-n_:]
                l = jnp.zeros_like(fire_t)
                l_ = t - fire_t
                z = y[:n_] < F_b
                z_ = out[:n_] >= F_b
                df = jnp.where(jnp.logical_and(z, z_), l_, l)
                fire_t_ = fire_t + df
                out = jnp.concatenate([out[:-n_], fire_t_], axis=0)

            should_sample = jnp.logical_and(
                jnp.equal((step_index + 1) % sample_stride, 0),
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

        carry = (y0, samples, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        (final_state, sampled, _, _), _ = jax.lax.scan(scan_func, carry, (t[:-1], time_delta_grid))
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


def odeint_sampled(func, y0, t, n_, F_b, sample_stride, sample_neurons):
    t = jnp.asarray(t, dtype=JAX_DTYPE)
    y0 = jnp.asarray(y0, dtype=JAX_DTYPE)
    jax_check_type(t, y0)
    F_b = jnp.asarray(F_b, dtype=y0.dtype)
    return _JaxIntegrator(n_, F_b).integrate_sampled(func, y0, t, sample_stride, sample_neurons)