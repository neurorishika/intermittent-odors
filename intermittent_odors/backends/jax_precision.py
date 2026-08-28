import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

_ALIASES = {
    'float64': 'float64',
    'x64': 'float64',
    'float32': 'float32',
    'x32': 'float32',
    'bfloat16': 'bfloat16',
    'bf16': 'bfloat16',
}

#: Opt out of the reduced-precision guards below. Set by the benchmark harness,
#: which measures these modes deliberately.
ALLOW_REDUCED_PRECISION = os.environ.get('IODOR_ALLOW_REDUCED_PRECISION', '').strip().lower() in {'1', 'true', 'yes'}

PRECISION_MODE = _ALIASES.get(os.environ.get('IODOR_JAX_PRECISION', 'float64').strip().lower())
if PRECISION_MODE is None:
    supported = ', '.join(sorted(_ALIASES))
    raise ValueError(f'Unsupported IODOR_JAX_PRECISION value. Expected one of: {supported}.')

# Measured on the realistic-slurm case (see docs/performance.md): bfloat16 diverges to
# non-finite values on both CPU and GPU, and float32 survives a 200 ms trial to ~1e-05
# but goes non-finite on CPU by 1000 ms. Neither is safe for production rollouts, so
# neither is allowed to be selected silently.
if PRECISION_MODE == 'bfloat16' and not ALLOW_REDUCED_PRECISION:
    raise ValueError(
        'IODOR_JAX_PRECISION=bfloat16 produces non-finite output for this model on both '
        'CPU and GPU and must not be used for production runs. Set '
        'IODOR_ALLOW_REDUCED_PRECISION=1 to override for benchmarking.'
    )

if PRECISION_MODE == 'float32' and not ALLOW_REDUCED_PRECISION:
    warnings.warn(
        'IODOR_JAX_PRECISION=float32 diverges from the float64 baseline for this model: '
        'non-finite on CPU at 1000 ms rollouts, and ~16 mV from baseline on GPU. Validate '
        'at your actual rollout length and judge on task metrics, not max_abs_diff. Set '
        'IODOR_ALLOW_REDUCED_PRECISION=1 to silence this warning.',
        RuntimeWarning,
        stacklevel=2,
    )

if PRECISION_MODE == 'float64':
    jax.config.update('jax_enable_x64', True)
    JAX_DTYPE = jnp.float64
    NP_DTYPE = np.float64
elif PRECISION_MODE == 'float32':
    jax.config.update('jax_enable_x64', False)
    JAX_DTYPE = jnp.float32
    NP_DTYPE = np.float32
else:
    jax.config.update('jax_enable_x64', False)
    JAX_DTYPE = jnp.bfloat16
    NP_DTYPE = np.float32


def to_numpy_dtype(value):
    return np.asarray(value, dtype=NP_DTYPE)


def to_jax_dtype(value):
    return jnp.asarray(np.asarray(value, dtype=NP_DTYPE), dtype=JAX_DTYPE)
