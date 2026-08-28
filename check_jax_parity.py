import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_shared_tf_parity import build_case, build_repo_production_case
from check_shared_tf_parity import evaluate as evaluate_tf
from intermittent_odors.backends.jax_integrator import odeint as jax_odeint
from intermittent_odors.backends.jax_network import \
    build_dynamics as build_jax_dynamics
from intermittent_odors.backends.tf_network import \
    build_dynamics as build_tf_dynamics


def max_abs_diff(left, right):
    matching_special = (
        (np.isnan(left) & np.isnan(right))
        | (np.isposinf(left) & np.isposinf(right))
        | (np.isneginf(left) & np.isneginf(right))
    )
    valid = ~matching_special
    if not np.any(valid):
        return 0.0
    diff = np.abs(np.nan_to_num(left[valid] - right[valid], nan=np.inf, posinf=np.inf, neginf=np.inf))
    return float(np.max(diff))


def evaluate_jax(config, current_input, state, times, thresholds):
    dXdt = build_jax_dynamics(config, current_input)
    state_tensor = jnp.asarray(np.asarray(state, dtype=np.float64), dtype=jnp.float64)
    times_tensor = jnp.asarray(np.asarray(times, dtype=np.float64), dtype=jnp.float64)
    derivative = np.asarray(dXdt(state_tensor, times_tensor[1]))
    rollout = np.asarray(jax_odeint(dXdt, state_tensor, times_tensor, config['n_n'], thresholds))
    return derivative, rollout


def run_case(name, config, current_input, state, times, thresholds, atol=1e-10, rtol=1e-10):
    tf_derivative, tf_rollout = evaluate_tf(build_tf_dynamics, config, current_input, state, times, thresholds)
    jax_derivative, jax_rollout = evaluate_jax(config, current_input, state, times, thresholds)

    derivative_close = np.allclose(tf_derivative, jax_derivative, atol=atol, rtol=rtol, equal_nan=True)
    rollout_close = np.allclose(tf_rollout, jax_rollout, atol=atol, rtol=rtol, equal_nan=True)

    print(f'{name}: derivative max abs diff = {max_abs_diff(tf_derivative, jax_derivative):.3e}')
    print(f'{name}: rollout max abs diff = {max_abs_diff(tf_rollout, jax_rollout):.3e}')

    if not derivative_close or not rollout_close:
        raise SystemExit(f'JAX parity check failed for {name}')


def main():
    run_case('production_like', *build_case(seed=1, n_n=6, p_n=4, ach_density=0.35, fgaba_density=0.45, sgaba_density=0.25))
    run_case('ln_only_like', *build_case(seed=2, n_n=5, p_n=1, ach_density=0.0, fgaba_density=0.5, sgaba_density=0.0))
    run_case('repo_production_topology', *build_repo_production_case(graph_no=1, odor_seed=59428, trial_seed=1))
    print('JAX shared core parity checks passed.')


if __name__ == '__main__':
    main()