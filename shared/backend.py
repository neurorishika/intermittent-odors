import hashlib
import os

import numpy as np

_BACKEND_ALIASES = {
    'jax': 'jax',
    'tf': 'tensorflow',
    'tensorflow': 'tensorflow',
}

_JAX_INTEGRATOR_CACHE = {}
_JAX_BATCH_INTEGRATOR_CACHE = {}
_JAX_SAMPLED_INTEGRATOR_CACHE = {}
_JAX_BATCH_SAMPLED_INTEGRATOR_CACHE = {}
_JAX_CACHE_CONFIGURED = False


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


def integrate_trajectory_batch(config, current_inputs, state_vectors, times, thresholds, backend=None):
    backend_name = get_backend_name(backend)
    if backend_name == 'jax':
        return _integrate_jax_batch(config, current_inputs, state_vectors, times, thresholds)

    outputs = [
        _integrate_tensorflow(config, current_input, state_vector, times, thresholds)
        for current_input, state_vector in zip(current_inputs, state_vectors)
    ]
    return np.asarray(outputs, dtype=np.float64)


def integrate_trajectory_sampled(config, current_input, state_vector, times, thresholds, sample_stride, sample_neurons=None, backend=None):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    if backend_name == 'jax':
        return _integrate_jax_sampled(
            config,
            current_input,
            state_vector,
            times,
            thresholds,
            int(sample_stride),
            sample_neurons,
        )

    trajectory = _integrate_tensorflow(config, current_input, state_vector, times, thresholds)
    sampled = np.asarray(trajectory[:-1:int(sample_stride), :sample_neurons], dtype=np.float64)
    final_state = np.asarray(trajectory[-1], dtype=np.float64)
    return sampled, final_state


def integrate_trajectory_sampled_batch(config, current_inputs, state_vectors, times, thresholds, sample_stride, sample_neurons=None, backend=None):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    if backend_name == 'jax':
        return _integrate_jax_sampled_batch(
            config,
            current_inputs,
            state_vectors,
            times,
            thresholds,
            int(sample_stride),
            sample_neurons,
        )

    trajectory = integrate_trajectory_batch(config, current_inputs, state_vectors, times, thresholds, backend=backend_name)
    sampled = np.asarray(trajectory[:, :-1:int(sample_stride), :sample_neurons], dtype=np.float64)
    final_state = np.asarray(trajectory[:, -1, :], dtype=np.float64)
    return sampled, final_state


def get_sampled_integrator_runner(config, current_input, thresholds, sample_stride, sample_neurons=None, backend=None):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    if backend_name == 'jax':
        return _get_jax_sampled_runner(config, current_input, thresholds, int(sample_stride), sample_neurons)

    def run(state_vector, times):
        return integrate_trajectory_sampled(
            config,
            current_input,
            state_vector,
            times,
            thresholds,
            sample_stride,
            sample_neurons=sample_neurons,
            backend=backend_name,
        )

    return run


def get_sampled_integrator_runner_batch(config, current_inputs, thresholds, sample_stride, sample_neurons=None, backend=None):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    if backend_name == 'jax':
        return _get_jax_sampled_batch_runner(config, current_inputs, thresholds, int(sample_stride), sample_neurons)

    def run(state_vectors, times):
        return integrate_trajectory_sampled_batch(
            config,
            current_inputs,
            state_vectors,
            times,
            thresholds,
            sample_stride,
            sample_neurons=sample_neurons,
            backend=backend_name,
        )

    return run


def _integrate_jax(config, current_input, state_vector, times, thresholds):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_integrator(config, current_input)
    return np.asarray(
        compiled(
            np.asarray(state_vector, dtype=np_dtype),
            np.asarray(current_input, dtype=np_dtype).T,
            np.asarray(times, dtype=np_dtype),
            np.asarray(thresholds, dtype=np_dtype),
        ),
        dtype=np.float64,
    )


def _integrate_jax_batch(config, current_inputs, state_vectors, times, thresholds):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_batch_integrator(config, current_inputs)
    return np.asarray(
        compiled(
            np.asarray(state_vectors, dtype=np_dtype),
            np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1),
            np.asarray(times, dtype=np_dtype),
            np.asarray(thresholds, dtype=np_dtype),
        ),
        dtype=np.float64,
    )


def _integrate_jax_sampled(config, current_input, state_vector, times, thresholds, sample_stride, sample_neurons):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_integrator(config, current_input, sample_stride, sample_neurons)
    sampled, final_state = compiled(
        np.asarray(state_vector, dtype=np_dtype),
        np.asarray(current_input, dtype=np_dtype).T,
        np.asarray(times, dtype=np_dtype),
        np.asarray(thresholds, dtype=np_dtype),
    )
    return np.asarray(sampled, dtype=np.float64), np.asarray(final_state, dtype=np.float64)


def _integrate_jax_sampled_batch(config, current_inputs, state_vectors, times, thresholds, sample_stride, sample_neurons):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_batch_integrator(config, current_inputs, sample_stride, sample_neurons)
    sampled, final_state = compiled(
        np.asarray(state_vectors, dtype=np_dtype),
        np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1),
        np.asarray(times, dtype=np_dtype),
        np.asarray(thresholds, dtype=np_dtype),
    )
    return np.asarray(sampled, dtype=np.float64), np.asarray(final_state, dtype=np.float64)


def _get_jax_sampled_runner(config, current_input, thresholds, sample_stride, sample_neurons):
    import jax

    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_integrator(config, current_input, sample_stride, sample_neurons)
    current_input_tensor = jax.device_put(np.asarray(current_input, dtype=np_dtype).T)
    fire_thresholds = jax.device_put(np.asarray(thresholds, dtype=np_dtype))

    def run(state_vector, times):
        return compiled(state_vector, current_input_tensor, times, fire_thresholds)

    return run


def _get_jax_sampled_batch_runner(config, current_inputs, thresholds, sample_stride, sample_neurons):
    import jax

    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_batch_integrator(config, current_inputs, sample_stride, sample_neurons)
    current_input_tensor = jax.device_put(np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1))
    fire_thresholds = jax.device_put(np.asarray(thresholds, dtype=np_dtype))

    def run(state_vectors, times):
        return compiled(state_vectors, current_input_tensor, times, fire_thresholds)

    return run


def _get_compiled_jax_integrator(config, current_input):
    cache_key = _fingerprint_jax_problem(config, current_input)
    compiled = _JAX_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from shared.jax_integrator import odeint
    from shared.jax_network import build_dynamics_core

    dXdt = build_dynamics_core(config)
    n_n = int(config['n_n'])
    compiled = jax.jit(
        lambda state, current_input_tensor, trajectory_times, fire_thresholds: odeint(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
        )
    )
    _JAX_INTEGRATOR_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_jax_batch_integrator(config, current_inputs):
    cache_key = _fingerprint_jax_problem(config, current_inputs)
    compiled = _JAX_BATCH_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from shared.jax_integrator import odeint
    from shared.jax_network import build_dynamics_core

    dXdt = build_dynamics_core(config)
    n_n = int(config['n_n'])

    def integrate_single(state, current_input_tensor, trajectory_times, fire_thresholds):
        return odeint(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
        )

    compiled = jax.jit(jax.vmap(integrate_single, in_axes=(0, 0, None, None)))
    _JAX_BATCH_INTEGRATOR_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_jax_sampled_integrator(config, current_input, sample_stride, sample_neurons):
    cache_key = f"{_fingerprint_jax_problem(config, current_input)}|sample_stride={sample_stride}|sample_neurons={sample_neurons}"
    compiled = _JAX_SAMPLED_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from shared.jax_integrator import odeint_sampled
    from shared.jax_network import build_dynamics_core

    dXdt = build_dynamics_core(config)
    n_n = int(config['n_n'])
    compiled = jax.jit(
        lambda state, current_input_tensor, trajectory_times, fire_thresholds: odeint_sampled(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
            sample_stride,
            sample_neurons,
        ),
        donate_argnums=(0,),
    )
    _JAX_SAMPLED_INTEGRATOR_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_jax_sampled_batch_integrator(config, current_inputs, sample_stride, sample_neurons):
    cache_key = f"{_fingerprint_jax_problem(config, current_inputs)}|sample_stride={sample_stride}|sample_neurons={sample_neurons}|batched=1"
    compiled = _JAX_BATCH_SAMPLED_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from shared.jax_integrator import odeint_sampled
    from shared.jax_network import build_dynamics_core

    dXdt = build_dynamics_core(config)
    n_n = int(config['n_n'])

    def integrate_single(state, current_input_tensor, trajectory_times, fire_thresholds):
        return odeint_sampled(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
            sample_stride,
            sample_neurons,
        )

    compiled = jax.jit(
        jax.vmap(integrate_single, in_axes=(0, 0, None, None)),
        donate_argnums=(0,),
    )
    _JAX_BATCH_SAMPLED_INTEGRATOR_CACHE[cache_key] = compiled
    return compiled


def _configure_jax_compilation_cache():
    global _JAX_CACHE_CONFIGURED
    if _JAX_CACHE_CONFIGURED:
        return

    cache_dir = os.environ.get('IODOR_JAX_COMPILATION_CACHE_DIR') or os.environ.get('JAX_COMPILATION_CACHE_DIR')
    if cache_dir:
        import jax

        jax.config.update('jax_compilation_cache_dir', cache_dir)
        jax.config.update(
            'jax_persistent_cache_min_compile_time_secs',
            float(os.environ.get('IODOR_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', 0)),
        )
        jax.config.update(
            'jax_persistent_cache_min_entry_size_bytes',
            int(os.environ.get('IODOR_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', -1)),
        )

    _JAX_CACHE_CONFIGURED = True


def _fingerprint_jax_problem(config, current_input):
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_get_jax_precision_mode().encode('utf-8'))
    for key in sorted(config):
        hasher.update(key.encode('utf-8'))
        _hash_array_signature(hasher, config[key])
    hasher.update(b'current_input')
    _hash_array_signature(hasher, current_input)
    return hasher.hexdigest()


def _hash_array_signature(hasher, value):
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str(array.dtype).encode('utf-8'))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())


def _get_jax_numpy_dtype():
    from shared.jax_precision import NP_DTYPE

    return NP_DTYPE


def _get_jax_precision_mode():
    from shared.jax_precision import PRECISION_MODE

    return PRECISION_MODE


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