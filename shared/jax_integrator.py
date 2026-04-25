import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


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

    def _step_func(self, func, t, dt, y):
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = jnp.asarray(dt, dtype=y.dtype)

        k2 = func(y + dt_cast * k1 / 2, half_step)
        k3 = func(y + dt_cast * k2 / 2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        return (k1 + 2 * k2 + 2 * k3 + k4) * (dt_cast / 6)


def odeint(func, y0, t, n_, F_b):
    t = jnp.asarray(np.asarray(t, dtype=np.float64), dtype=jnp.float64)
    y0 = jnp.asarray(y0)
    jax_check_type(t, y0)
    F_b = jnp.asarray(np.asarray(F_b, dtype=np.float64), dtype=y0.dtype)
    return _JaxIntegrator(n_, F_b).integrate(func, y0, t)