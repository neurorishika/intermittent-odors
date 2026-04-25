from dataclasses import dataclass, field
from hashlib import blake2b
from types import MappingProxyType

import numpy as np

DEFAULT_INPUT_DT = 0.01


@dataclass(frozen=True)
class NetworkSpec:
    n_n: int
    p_n: int
    l_n: int
    ach_mat: np.ndarray
    fgaba_mat: np.ndarray
    sgaba_mat: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'n_n', int(self.n_n))
        object.__setattr__(self, 'p_n', int(self.p_n))
        object.__setattr__(self, 'l_n', int(self.l_n))
        object.__setattr__(self, 'ach_mat', np.ascontiguousarray(np.asarray(self.ach_mat, dtype=np.float64)))
        object.__setattr__(self, 'fgaba_mat', np.ascontiguousarray(np.asarray(self.fgaba_mat, dtype=np.float64)))
        object.__setattr__(self, 'sgaba_mat', np.ascontiguousarray(np.asarray(self.sgaba_mat, dtype=np.float64)))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class StimulusSpec:
    current_input: np.ndarray
    state_vector: np.ndarray
    times: np.ndarray
    thresholds: np.ndarray
    input_dt: float | None = None
    sample_stride: int = 1
    sample_neurons: int | None = None
    time_batches: tuple[np.ndarray, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        times = np.ascontiguousarray(np.asarray(self.times, dtype=np.float64))
        time_batches = tuple(
            np.ascontiguousarray(np.asarray(batch, dtype=np.float64))
            for batch in self.time_batches
        )
        input_dt = self.input_dt
        if input_dt is None:
            if time_batches:
                input_dt = infer_input_dt_from_batches(time_batches)
            else:
                input_dt = infer_input_dt_from_times(times)

        object.__setattr__(self, 'current_input', np.ascontiguousarray(np.asarray(self.current_input, dtype=np.float64)))
        object.__setattr__(self, 'state_vector', np.ascontiguousarray(np.asarray(self.state_vector, dtype=np.float64)))
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'thresholds', np.ascontiguousarray(np.asarray(self.thresholds, dtype=np.float64)))
        object.__setattr__(self, 'input_dt', float(input_dt))
        object.__setattr__(self, 'sample_stride', int(self.sample_stride))
        sample_neurons = self.sample_neurons
        if sample_neurons is not None:
            sample_neurons = int(sample_neurons)
        object.__setattr__(self, 'sample_neurons', sample_neurons)
        object.__setattr__(self, 'time_batches', time_batches)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ExperimentSpec:
    config: dict
    network: NetworkSpec
    stimulus: StimulusSpec
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'config', MappingProxyType(_normalize_config(self.config, input_dt=self.stimulus.input_dt)))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    def prepare(self, **overrides):
        prepared = PreparedExperiment(
            config=dict(self.config),
            thresholds=self.stimulus.thresholds,
            input_dt=self.stimulus.input_dt,
            sample_stride=self.stimulus.sample_stride,
            sample_neurons=self.stimulus.sample_neurons or self.network.n_n,
            time_batches=self.stimulus.time_batches,
            metadata={
                'network': dict(self.network.metadata),
                'stimulus': dict(self.stimulus.metadata),
                **dict(self.metadata),
            },
        )
        if overrides:
            prepared = prepared.with_overrides(**overrides)
        return prepared

    @property
    def current_input(self):
        return self.stimulus.current_input

    @property
    def state_vector(self):
        return self.stimulus.state_vector

    @property
    def times(self):
        return self.stimulus.times

    @property
    def thresholds(self):
        return self.stimulus.thresholds

    @property
    def time_batches(self):
        return self.stimulus.time_batches

    @property
    def sample_stride(self):
        return self.stimulus.sample_stride

    @property
    def sample_neurons(self):
        return self.stimulus.sample_neurons or self.network.n_n


@dataclass(frozen=True)
class PreparedExperiment:
    config: dict
    thresholds: np.ndarray
    input_dt: float = DEFAULT_INPUT_DT
    sample_stride: int = 1
    sample_neurons: int | None = None
    time_batches: tuple[np.ndarray, ...] = ()
    metadata: dict = field(default_factory=dict)
    model_digest: str = field(init=False)

    def __post_init__(self):
        normalized_config = _normalize_config(self.config, input_dt=self.input_dt)
        thresholds = np.ascontiguousarray(np.asarray(self.thresholds, dtype=np.float64))
        sample_stride = int(self.sample_stride)
        sample_neurons = int(self.sample_neurons or normalized_config['n_n'])
        time_batches = tuple(
            np.ascontiguousarray(np.asarray(batch, dtype=np.float64))
            for batch in self.time_batches
        )

        object.__setattr__(self, 'config', MappingProxyType(normalized_config))
        object.__setattr__(self, 'thresholds', thresholds)
        object.__setattr__(self, 'input_dt', float(normalized_config['input_dt']))
        object.__setattr__(self, 'sample_stride', sample_stride)
        object.__setattr__(self, 'sample_neurons', sample_neurons)
        object.__setattr__(self, 'time_batches', time_batches)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, 'model_digest', _hash_prepared_config(normalized_config))

    @property
    def n_n(self):
        return int(self.config['n_n'])

    @property
    def p_n(self):
        return int(self.config['p_n'])

    @property
    def l_n(self):
        return int(self.config['l_n'])

    def with_overrides(self, *, input_dt=None, sample_stride=None, sample_neurons=None, time_batches=None, metadata=None):
        return PreparedExperiment(
            config=dict(self.config),
            thresholds=self.thresholds,
            input_dt=self.input_dt if input_dt is None else float(input_dt),
            sample_stride=self.sample_stride if sample_stride is None else int(sample_stride),
            sample_neurons=self.sample_neurons if sample_neurons is None else int(sample_neurons),
            time_batches=self.time_batches if time_batches is None else tuple(time_batches),
            metadata=dict(self.metadata) if metadata is None else metadata,
        )


def prepare_experiment(config, thresholds, *, input_dt=None, sample_stride=1, sample_neurons=None, time_batches=None, metadata=None):
    return PreparedExperiment(
        config=config,
        thresholds=thresholds,
        input_dt=DEFAULT_INPUT_DT if input_dt is None else float(input_dt),
        sample_stride=int(sample_stride),
        sample_neurons=sample_neurons,
        time_batches=() if time_batches is None else tuple(time_batches),
        metadata={} if metadata is None else metadata,
    )


def build_network_spec(*, n_n, p_n, l_n, ach_mat, fgaba_mat, sgaba_mat, metadata=None):
    return NetworkSpec(
        n_n=n_n,
        p_n=p_n,
        l_n=l_n,
        ach_mat=ach_mat,
        fgaba_mat=fgaba_mat,
        sgaba_mat=sgaba_mat,
        metadata={} if metadata is None else metadata,
    )


def build_network_spec_from_config(config, metadata=None):
    config = dict(config)
    return build_network_spec(
        n_n=config['n_n'],
        p_n=config['p_n'],
        l_n=config['l_n'],
        ach_mat=config['ach_mat'],
        fgaba_mat=config['fgaba_mat'],
        sgaba_mat=config['sgaba_mat'],
        metadata={} if metadata is None else metadata,
    )


def build_stimulus_spec(current_input, state_vector, times, thresholds, *, input_dt=None, sample_stride=1, sample_neurons=None, time_batches=None, metadata=None):
    return StimulusSpec(
        current_input=current_input,
        state_vector=state_vector,
        times=times,
        thresholds=thresholds,
        input_dt=input_dt,
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
        time_batches=() if time_batches is None else tuple(time_batches),
        metadata={} if metadata is None else metadata,
    )


def build_experiment_spec(config, current_input, state_vector, times, thresholds, *, input_dt=None, sample_stride=1, sample_neurons=None, time_batches=None, metadata=None, network_metadata=None, stimulus_metadata=None):
    network = build_network_spec_from_config(config, metadata=network_metadata)
    stimulus = build_stimulus_spec(
        current_input,
        state_vector,
        times,
        thresholds,
        input_dt=input_dt,
        sample_stride=sample_stride,
        sample_neurons=sample_neurons or int(network.n_n),
        time_batches=time_batches,
        metadata=stimulus_metadata,
    )
    return ExperimentSpec(
        config=config,
        network=network,
        stimulus=stimulus,
        metadata={} if metadata is None else metadata,
    )


def ensure_prepared_experiment(experiment_or_config, *, thresholds=None, input_dt=None, sample_stride=None, sample_neurons=None, time_batches=None, metadata=None):
    if isinstance(experiment_or_config, ExperimentSpec):
        return experiment_or_config.prepare(
            input_dt=input_dt,
            sample_stride=sample_stride,
            sample_neurons=sample_neurons,
            time_batches=time_batches,
            metadata=metadata,
        )

    if isinstance(experiment_or_config, PreparedExperiment):
        if all(value is None for value in (input_dt, sample_stride, sample_neurons, time_batches, metadata)):
            return experiment_or_config
        return experiment_or_config.with_overrides(
            input_dt=input_dt,
            sample_stride=sample_stride,
            sample_neurons=sample_neurons,
            time_batches=time_batches,
            metadata=metadata,
        )

    if thresholds is None:
        raise ValueError('thresholds are required when preparing a legacy config dictionary.')

    return prepare_experiment(
        experiment_or_config,
        thresholds,
        input_dt=input_dt,
        sample_stride=1 if sample_stride is None else sample_stride,
        sample_neurons=sample_neurons,
        time_batches=time_batches,
        metadata=metadata,
    )


def infer_input_dt_from_times(times, default=DEFAULT_INPUT_DT):
    times = np.asarray(times, dtype=np.float64)
    if times.size >= 2:
        return float(times[1] - times[0])
    return float(default)


def infer_input_dt_from_batches(time_batches, default=DEFAULT_INPUT_DT):
    for batch in time_batches:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.size >= 2:
            return float(batch[1] - batch[0])
    return float(default)


def _normalize_config(config, input_dt):
    normalized = {}
    for key, value in dict(config).items():
        normalized[key] = _normalize_config_value(value)

    normalized['n_n'] = int(normalized['n_n'])
    normalized['p_n'] = int(normalized['p_n'])
    normalized['l_n'] = int(normalized['l_n'])
    normalized['input_dt'] = float(normalized.get('input_dt', input_dt if input_dt is not None else DEFAULT_INPUT_DT))
    normalized['input_scale'] = float(normalized.get('input_scale', 1.0 / normalized['input_dt']))

    for prefix in ('ach', 'fgaba', 'sgaba'):
        _attach_synapse_layout(normalized, prefix)

    return normalized


def _normalize_config_value(value):
    if isinstance(value, np.ndarray):
        return _normalize_array(value)
    if isinstance(value, (list, tuple)):
        return _normalize_array(np.asarray(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_array(array):
    array = np.asarray(array)
    if array.dtype.kind in 'biu':
        return np.ascontiguousarray(array.astype(np.int64, copy=False))
    if array.dtype.kind == 'f':
        return np.ascontiguousarray(array.astype(np.float64, copy=False))
    return np.ascontiguousarray(array)


def _attach_synapse_layout(config, prefix):
    matrix_key = f'{prefix}_mat'
    matrix = np.ascontiguousarray(np.asarray(config[matrix_key], dtype=np.float64))
    flat_indices = np.flatnonzero(matrix.reshape(-1) != 0.0).astype(np.int32, copy=False)
    config[matrix_key] = matrix
    config[f'{prefix}_indices'] = np.ascontiguousarray(flat_indices)
    config[f'{prefix}_row_ids'] = np.ascontiguousarray((flat_indices // config['n_n']).astype(np.int32, copy=False))
    config[f'{prefix}_col_ids'] = np.ascontiguousarray((flat_indices % config['n_n']).astype(np.int32, copy=False))
    config[f'n_syn_{prefix}'] = int(flat_indices.size)


def _hash_prepared_config(config):
    hasher = blake2b(digest_size=20)
    for key in sorted(config):
        hasher.update(key.encode('utf-8'))
        _hash_value(hasher, config[key])
    return hasher.hexdigest()


def _hash_value(hasher, value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode('utf-8'))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
        return

    if isinstance(value, (str, bytes)):
        hasher.update(str(value).encode('utf-8'))
        return

    if isinstance(value, (int, float, bool)):
        hasher.update(repr(value).encode('utf-8'))
        return

    hasher.update(repr(value).encode('utf-8'))