import os

import numpy as np

_BACKEND_ALIASES = {
    'jax': 'jax',
    'tf': 'tensorflow',
    'tensorflow': 'tensorflow',
}


def get_backend_name(backend=None):
    raw_backend = backend or os.environ.get('IODOR_BACKEND', 'tensorflow')
    canonical_backend = _BACKEND_ALIASES.get(raw_backend.strip().lower())
    if canonical_backend is None:
        supported = ', '.join(sorted(_BACKEND_ALIASES))
        raise ValueError(f'Unsupported backend {raw_backend!r}. Expected one of: {supported}.')
    return canonical_backend


def integrate_trajectory(config, current_input, state_vector, times, thresholds, backend=None):
    backend_name = get_backend_name(backend)
    if backend_name == 'jax':
        return _integrate_jax(config, current_input, state_vector, times, thresholds)
    return _integrate_tensorflow(config, current_input, state_vector, times, thresholds)


def _integrate_jax(config, current_input, state_vector, times, thresholds):
    from shared.jax_integrator import odeint
    from shared.jax_network import build_dynamics

    dXdt = build_dynamics(config, current_input)
    return np.asarray(
        odeint(
            dXdt,
            np.asarray(state_vector, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
            int(config['n_n']),
            thresholds,
        ),
        dtype=np.float64,
    )


def _integrate_tensorflow(config, current_input, state_vector, times, thresholds):
    import tensorflow.compat.v1 as tf

    from shared.tf_integrator import odeint
    from shared.tf_network import build_dynamics

    with tf.Graph().as_default():
        tf.disable_v2_behavior()
        dXdt = build_dynamics(config, current_input)
        init_state = tf.constant(np.asarray(state_vector, dtype=np.float64), dtype=tf.float64)
        tensor_state = odeint(
            dXdt,
            init_state,
            np.asarray(times, dtype=np.float64),
            int(config['n_n']),
            thresholds,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return np.asarray(sess.run(tensor_state), dtype=np.float64)