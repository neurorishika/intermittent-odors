import hashlib
import os

import numpy as np

from intermittent_odors.experiment import (PreparedExperiment,
                                           ensure_prepared_experiment,
                                           infer_input_dt_from_times)

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


def sample_phase_for_batch(time_batch, sample_stride):
    """Local step index of the first sample this time batch should emit.

    Time batches are chained by carrying the previous batch's final state, and
    ``slurm/builders.py`` overlaps them by one timepoint so no integration step
    is skipped at the seam. That overlap shifts a batch off the global sampling
    grid, so each batch is told where the grid falls inside it. Batches that
    start on a grid point get phase 0 and emit their initial state, exactly as
    an unbatched rollout does.
    """
    stride = int(sample_stride)
    if stride <= 1:
        return 0

    times = np.asarray(time_batch, dtype=np.float64)
    if times.shape[0] < 2:
        return 0

    dt = float(times[1] - times[0])
    if dt <= 0:
        return 0

    start_index = int(round(float(times[0]) / dt))
    return (-start_index) % stride


def sampled_length_for_batch(time_batch, sample_stride):
    """Number of sampled rows ``run_time_batches`` will emit for one batch."""
    stride = int(sample_stride)
    total_steps = int(np.asarray(time_batch).shape[0]) - 1
    if total_steps <= 0:
        return 1
    phase = sample_phase_for_batch(time_batch, stride)
    return max(0, -(-(total_steps - phase) // stride))


def get_backend_name(backend=None):
    raw_backend = backend or os.environ.get('IODOR_BACKEND', 'jax')
    canonical_backend = _BACKEND_ALIASES.get(raw_backend.strip().lower())
    if canonical_backend is None:
        supported = ', '.join(sorted(_BACKEND_ALIASES))
        raise ValueError(f'Unsupported backend {raw_backend!r}. Expected one of: {supported}.')
    return canonical_backend


class CompiledExperimentRunner:

    def __init__(self, experiment, backend=None):
        self.experiment = ensure_prepared_experiment(experiment)
        self.backend_name = get_backend_name(backend)

    def run(self, state_vector, current_input, times):
        if self.backend_name == 'jax':
            return _integrate_jax(self.experiment, current_input, state_vector, times)
        return _integrate_tensorflow(self.experiment, current_input, state_vector, times)

    def run_batch(self, state_vectors, current_inputs, times):
        if self.backend_name == 'jax':
            return _integrate_jax_batch(self.experiment, current_inputs, state_vectors, times)

        outputs = [
            _integrate_tensorflow(self.experiment, current_input, state_vector, times)
            for current_input, state_vector in zip(current_inputs, state_vectors)
        ]
        return np.asarray(outputs, dtype=np.float64)

    def run_sampled(self, state_vector, current_input, times, sample_phase=0):
        if self.backend_name == 'jax':
            return _integrate_jax_sampled(
                self.experiment, current_input, state_vector, times, sample_phase
            )

        trajectory = _integrate_tensorflow(self.experiment, current_input, state_vector, times)
        stride = int(self.experiment.sample_stride)
        phase = int(sample_phase) % stride
        sampled = np.asarray(
            trajectory[phase:-1:stride, :self.experiment.sample_neurons],
            dtype=np.float64,
        )
        final_state = np.asarray(trajectory[-1], dtype=np.float64)
        return sampled, final_state

    def run_sampled_batch(self, state_vectors, current_inputs, times, sample_phase=0):
        if self.backend_name == 'jax':
            return _integrate_jax_sampled_batch(
                self.experiment, current_inputs, state_vectors, times, sample_phase
            )

        trajectory = self.run_batch(state_vectors, current_inputs, times)
        stride = int(self.experiment.sample_stride)
        phase = int(sample_phase) % stride
        sampled = np.asarray(
            trajectory[:, phase:-1:stride, :self.experiment.sample_neurons],
            dtype=np.float64,
        )
        final_state = np.asarray(trajectory[:, -1, :], dtype=np.float64)
        return sampled, final_state

    def get_sampled_runner(self, current_input):
        if self.backend_name == 'jax':
            return _get_jax_sampled_runner(self.experiment, current_input)

        def run(state_vector, times, sample_phase=0):
            return self.run_sampled(state_vector, current_input, times, sample_phase)

        return run

    def get_sampled_batch_runner(self, current_inputs):
        if self.backend_name == 'jax':
            return _get_jax_sampled_batch_runner(self.experiment, current_inputs)

        def run(state_vectors, times, sample_phase=0):
            return self.run_sampled_batch(state_vectors, current_inputs, times, sample_phase)

        return run

    def run_time_batches(self, state_vector, current_input, time_batches=None, progress=None):
        time_batches = self.experiment.time_batches if time_batches is None else tuple(time_batches)
        runner = self.get_sampled_runner(current_input)
        sample_phases = [
            sample_phase_for_batch(time_batch, self.experiment.sample_stride)
            for time_batch in time_batches
        ]

        if self.backend_name == 'jax':
            import jax
            import jax.numpy as jnp

            from intermittent_odors.backends.jax_precision import NP_DTYPE

            state_vector = jax.device_put(np.asarray(state_vector, dtype=NP_DTYPE))
            prepared_time_batches = [
                jax.device_put(np.asarray(time_batch, dtype=NP_DTYPE))
                for time_batch in time_batches
            ]
        else:
            state_vector = np.asarray(state_vector, dtype=np.float64)
            prepared_time_batches = [np.asarray(time_batch, dtype=np.float64) for time_batch in time_batches]

        sampled_outputs = []
        iterator = enumerate(prepared_time_batches)
        if progress is not None:
            iterator = progress(iterator, total=len(prepared_time_batches))

        for index, time_batch in iterator:
            sampled, state_vector = runner(
                state_vector, time_batch, sample_phases[index]
            )
            sampled_outputs.append(sampled)

        if self.backend_name == 'jax':
            if sampled_outputs:
                output = np.asarray(jnp.concatenate(sampled_outputs, axis=0), dtype=np.float64)
            else:
                output = np.zeros((0, self.experiment.sample_neurons), dtype=np.float64)
            final_state = np.asarray(state_vector, dtype=np.float64)
            return output, final_state

        if sampled_outputs:
            output = np.concatenate(sampled_outputs, axis=0)
        else:
            output = np.zeros((0, self.experiment.sample_neurons), dtype=np.float64)
        return output, state_vector

    def run_time_batches_batch(self, state_vectors, current_inputs, time_batches=None, progress=None):
        time_batches = self.experiment.time_batches if time_batches is None else tuple(time_batches)
        batch_size = int(np.asarray(state_vectors).shape[0])
        runner = self.get_sampled_batch_runner(current_inputs)
        sample_phases = [
            sample_phase_for_batch(time_batch, self.experiment.sample_stride)
            for time_batch in time_batches
        ]

        if self.backend_name == 'jax':
            import jax
            import jax.numpy as jnp

            from intermittent_odors.backends.jax_precision import NP_DTYPE

            state_vectors = jax.device_put(np.asarray(state_vectors, dtype=NP_DTYPE))
            prepared_time_batches = [
                jax.device_put(np.asarray(time_batch, dtype=NP_DTYPE))
                for time_batch in time_batches
            ]
        else:
            state_vectors = np.asarray(state_vectors, dtype=np.float64)
            prepared_time_batches = [np.asarray(time_batch, dtype=np.float64) for time_batch in time_batches]

        sampled_outputs = []
        iterator = enumerate(prepared_time_batches)
        if progress is not None:
            iterator = progress(iterator, total=len(prepared_time_batches))

        for index, time_batch in iterator:
            sampled, state_vectors = runner(
                state_vectors, time_batch, sample_phases[index]
            )
            sampled_outputs.append(sampled)

        if self.backend_name == 'jax':
            if sampled_outputs:
                output = np.asarray(jnp.concatenate(sampled_outputs, axis=1), dtype=np.float64)
            else:
                output = np.zeros((batch_size, 0, self.experiment.sample_neurons), dtype=np.float64)
            final_state = np.asarray(state_vectors, dtype=np.float64)
            return output, final_state

        if sampled_outputs:
            output = np.concatenate(sampled_outputs, axis=1)
        else:
            output = np.zeros((batch_size, 0, self.experiment.sample_neurons), dtype=np.float64)
        return output, state_vectors


def compile_experiment(experiment, backend=None):
    return CompiledExperimentRunner(experiment, backend=backend)


def integrate_trajectory(config, current_input, state_vector, times, thresholds=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_times(times),
    )
    return compile_experiment(experiment, backend=backend).run(state_vector, current_input, times)


def integrate_trajectory_batch(config, current_inputs, state_vectors, times, thresholds=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_times(times),
    )
    return compile_experiment(experiment, backend=backend).run_batch(state_vectors, current_inputs, times)


def integrate_trajectory_sampled(config, current_input, state_vector, times, thresholds=None, sample_stride=1, sample_neurons=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_times(times),
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
    )
    return compile_experiment(experiment, backend=backend).run_sampled(state_vector, current_input, times)


def integrate_trajectory_sampled_batch(config, current_inputs, state_vectors, times, thresholds=None, sample_stride=1, sample_neurons=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_times(times),
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
    )
    return compile_experiment(experiment, backend=backend).run_sampled_batch(state_vectors, current_inputs, times)


def get_sampled_integrator_runner(config, current_input, thresholds=None, sample_stride=1, sample_neurons=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
    )
    return compile_experiment(experiment, backend=backend).get_sampled_runner(current_input)


def get_sampled_integrator_runner_batch(config, current_inputs, thresholds=None, sample_stride=1, sample_neurons=None, backend=None):
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
    )
    return compile_experiment(experiment, backend=backend).get_sampled_batch_runner(current_inputs)


def _integrate_jax(experiment, current_input, state_vector, times):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_integrator(experiment, current_input)
    return np.asarray(
        compiled(
            np.asarray(state_vector, dtype=np_dtype),
            np.asarray(current_input, dtype=np_dtype).T,
            np.asarray(times, dtype=np_dtype),
            np.asarray(experiment.thresholds, dtype=np_dtype),
        ),
        dtype=np.float64,
    )


def _integrate_jax_batch(experiment, current_inputs, state_vectors, times):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_batch_integrator(experiment, current_inputs)
    return np.asarray(
        compiled(
            np.asarray(state_vectors, dtype=np_dtype),
            np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1),
            np.asarray(times, dtype=np_dtype),
            np.asarray(experiment.thresholds, dtype=np_dtype),
        ),
        dtype=np.float64,
    )


def _integrate_jax_sampled(experiment, current_input, state_vector, times, sample_phase=0):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_integrator(experiment, current_input, sample_phase)
    sampled, final_state = compiled(
        np.asarray(state_vector, dtype=np_dtype),
        np.asarray(current_input, dtype=np_dtype).T,
        np.asarray(times, dtype=np_dtype),
        np.asarray(experiment.thresholds, dtype=np_dtype),
    )
    return np.asarray(sampled, dtype=np.float64), np.asarray(final_state, dtype=np.float64)


def _integrate_jax_sampled_batch(experiment, current_inputs, state_vectors, times, sample_phase=0):
    np_dtype = _get_jax_numpy_dtype()
    compiled = _get_compiled_jax_sampled_batch_integrator(experiment, current_inputs, sample_phase)
    sampled, final_state = compiled(
        np.asarray(state_vectors, dtype=np_dtype),
        np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1),
        np.asarray(times, dtype=np_dtype),
        np.asarray(experiment.thresholds, dtype=np_dtype),
    )
    return np.asarray(sampled, dtype=np.float64), np.asarray(final_state, dtype=np.float64)


def _get_jax_sampled_runner(experiment, current_input):
    import jax

    np_dtype = _get_jax_numpy_dtype()
    current_input_tensor = jax.device_put(np.asarray(current_input, dtype=np_dtype).T)
    fire_thresholds = jax.device_put(np.asarray(experiment.thresholds, dtype=np_dtype))

    def run(state_vector, times, sample_phase=0):
        compiled = _get_compiled_jax_sampled_integrator(experiment, current_input, sample_phase)
        return compiled(state_vector, current_input_tensor, times, fire_thresholds)

    return run


def _get_jax_sampled_batch_runner(experiment, current_inputs):
    import jax

    np_dtype = _get_jax_numpy_dtype()
    current_input_tensor = jax.device_put(np.asarray(current_inputs, dtype=np_dtype).transpose(0, 2, 1))
    fire_thresholds = jax.device_put(np.asarray(experiment.thresholds, dtype=np_dtype))

    def run(state_vectors, times, sample_phase=0):
        compiled = _get_compiled_jax_sampled_batch_integrator(
            experiment, current_inputs, sample_phase
        )
        return compiled(state_vectors, current_input_tensor, times, fire_thresholds)

    return run


def _get_compiled_jax_integrator(experiment, current_input):
    cache_key = _fingerprint_jax_problem(experiment, current_input)
    compiled = _JAX_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from intermittent_odors.backends.jax_integrator import odeint

    dXdt = _get_dynamics_fn(experiment)
    n_n = int(experiment.config['n_n'])
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


def _get_compiled_jax_batch_integrator(experiment, current_inputs):
    cache_key = _fingerprint_jax_problem(experiment, current_inputs)
    compiled = _JAX_BATCH_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from intermittent_odors.backends.jax_integrator import odeint

    dXdt = _get_dynamics_fn(experiment)
    n_n = int(experiment.config['n_n'])

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


def _get_compiled_jax_sampled_integrator(experiment, current_input, sample_phase=0):
    sample_phase = int(sample_phase) % int(experiment.sample_stride)
    cache_key = (
        f"{_fingerprint_jax_problem(experiment, current_input)}"
        f"|sample_stride={experiment.sample_stride}|sample_neurons={experiment.sample_neurons}"
        f"|sample_phase={sample_phase}"
    )
    compiled = _JAX_SAMPLED_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from intermittent_odors.backends.jax_integrator import odeint_sampled

    dXdt = _get_dynamics_fn(experiment)
    n_n = int(experiment.config['n_n'])
    compiled = jax.jit(
        lambda state, current_input_tensor, trajectory_times, fire_thresholds: odeint_sampled(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
            experiment.sample_stride,
            experiment.sample_neurons,
            sample_phase,
        ),
        donate_argnums=(0,),
    )
    _JAX_SAMPLED_INTEGRATOR_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_jax_sampled_batch_integrator(experiment, current_inputs, sample_phase=0):
    sample_phase = int(sample_phase) % int(experiment.sample_stride)
    cache_key = (
        f"{_fingerprint_jax_problem(experiment, current_inputs)}"
        f"|sample_stride={experiment.sample_stride}|sample_neurons={experiment.sample_neurons}|batched=1"
        f"|sample_phase={sample_phase}"
    )
    compiled = _JAX_BATCH_SAMPLED_INTEGRATOR_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    _configure_jax_compilation_cache()

    import jax

    from intermittent_odors.backends.jax_integrator import odeint_sampled

    dXdt = _get_dynamics_fn(experiment)
    n_n = int(experiment.config['n_n'])

    def integrate_single(state, current_input_tensor, trajectory_times, fire_thresholds):
        return odeint_sampled(
            lambda X, t: dXdt(X, t, current_input_tensor),
            state,
            trajectory_times,
            n_n,
            fire_thresholds,
            experiment.sample_stride,
            experiment.sample_neurons,
            sample_phase,
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


def _fingerprint_jax_problem(experiment, current_input):
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(_get_jax_precision_mode().encode('utf-8'))
    if isinstance(experiment, PreparedExperiment):
        hasher.update(experiment.model_digest.encode('utf-8'))
        if experiment.dynamics_builder is not None:
            # Prevent cache collisions with custom dynamics functions.
            # id() is stable within a session; cross-session JIT recompiles anyway.
            hasher.update(f'custom_dyn:{id(experiment.dynamics_builder)}'.encode('utf-8'))
    else:
        raise TypeError('Expected a PreparedExperiment when building a JAX cache key.')
    hasher.update(b'current_input')
    _hash_array_signature(hasher, current_input)
    return hasher.hexdigest()


def _get_dynamics_fn(experiment):
    """Return dXdt(X, t, input_tensor) for this experiment.

    For standard experiments ``dynamics_builder`` is None and we delegate to
    the unchanged ``build_dynamics_core`` in ``jax_network.py``.
    For custom network models the stored callable is used instead.
    """
    if getattr(experiment, 'dynamics_builder', None) is not None:
        return experiment.dynamics_builder(experiment.config)
    from intermittent_odors.backends.jax_network import build_dynamics_core
    return build_dynamics_core(experiment.config)


def _hash_array_signature(hasher, value):
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str(array.dtype).encode('utf-8'))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())


def _get_jax_numpy_dtype():
    from intermittent_odors.backends.jax_precision import NP_DTYPE

    return NP_DTYPE


def _get_jax_precision_mode():
    from intermittent_odors.backends.jax_precision import PRECISION_MODE

    return PRECISION_MODE


def _integrate_tensorflow(experiment, current_input, state_vector, times):
    import tensorflow.compat.v1 as tf

    from intermittent_odors.backends.tf_integrator import odeint
    from intermittent_odors.backends.tf_network import build_dynamics

    with tf.Graph().as_default():
        tf.disable_v2_behavior()
        dXdt = build_dynamics(experiment.config, current_input)
        init_state = tf.constant(np.asarray(state_vector, dtype=np.float64), dtype=tf.float64)
        tensor_state = odeint(
            dXdt,
            init_state,
            np.asarray(times, dtype=np.float64),
            int(experiment.config['n_n']),
            experiment.thresholds,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return np.asarray(sess.run(tensor_state), dtype=np.float64)