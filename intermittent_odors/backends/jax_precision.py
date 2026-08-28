import os

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

PRECISION_MODE = _ALIASES.get(os.environ.get('IODOR_JAX_PRECISION', 'float64').strip().lower())
if PRECISION_MODE is None:
    supported = ', '.join(sorted(_ALIASES))
    raise ValueError(f'Unsupported IODOR_JAX_PRECISION value. Expected one of: {supported}.')

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