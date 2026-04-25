import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrialSettings:
    n_n: int = 120
    p_n: int = 90
    l_n: int = 30
    p_pnpn: float = 0.0
    p_pnln: float = 0.1
    p_lnpn: float = 0.2
    blocktime: int = 12000
    buffer: int = 500
    sim_res: float = 0.01
    min_block: int = 50
    batch_ms: int = 1000
    switch_prob: float = 0.1


def load_trial_settings(env=None):
    env = env or {}
    return TrialSettings(
        blocktime=int(env.get('IODOR_BLOCKTIME_MS', 12000)),
        buffer=int(env.get('IODOR_BUFFER_MS', 500)),
        sim_res=float(env.get('IODOR_SIM_RES_MS', 0.01)),
        min_block=int(env.get('IODOR_MIN_BLOCK_MS', 50)),
        batch_ms=int(env.get('IODOR_BATCH_MS', 1000)),
        switch_prob=float(env.get('IODOR_SWITCH_PROB', 0.1)),
    )


def build_trial_case(graph_no, odor_seed, trial_seed, network_dir, settings, deterministic_staging=False):
    ach_mat, fgaba_mat, sgaba_mat = build_connectivity_matrices(graph_no, network_dir, settings)
    current_input = build_current_input(graph_no, odor_seed, trial_seed, settings, deterministic_staging)
    state_vector = build_state_vector(
        graph_no,
        odor_seed,
        trial_seed,
        settings,
        int(np.sum(ach_mat)),
        int(np.sum(fgaba_mat)),
        int(np.sum(sgaba_mat)),
        deterministic_staging,
    )
    times = build_timepoints(settings)
    time_batches = split_timepoints(times, settings)
    return {
        'ach_mat': ach_mat,
        'fgaba_mat': fgaba_mat,
        'sgaba_mat': sgaba_mat,
        'current_input': current_input,
        'state_vector': state_vector,
        'times': times,
        'time_batches': time_batches,
    }


def build_connectivity_matrices(graph_no, network_dir, settings):
    ach_mat = np.zeros((settings.n_n, settings.n_n))
    ach_rng = np.random.RandomState(64163 + graph_no)
    ach_mat[settings.p_n:, :settings.p_n] = ach_rng.choice(
        [0.0, 1.0],
        size=(settings.l_n, settings.p_n),
        p=(1 - settings.p_pnln, settings.p_pnln),
    )
    ach_mat[:settings.p_n, :settings.p_n] = ach_rng.choice(
        [0.0, 1.0],
        size=(settings.p_n, settings.p_n),
        p=(1 - settings.p_pnpn, settings.p_pnpn),
    )

    lnpn = np.zeros((settings.p_n, settings.l_n))
    stride = int(settings.p_n / settings.l_n)
    spread = (round(settings.p_lnpn * settings.p_n) // 2) * 2 + 1
    center = 0
    index = np.arange(settings.p_n)
    for column in range(settings.l_n):
        idx = index[np.arange(center - spread // 2, 1 + center + spread // 2) % settings.p_n]
        lnpn[idx, column] = 1
        center += stride

    fgaba_mat = np.zeros((settings.n_n, settings.n_n))
    fgaba_mat[:settings.p_n, settings.p_n:] = lnpn
    fgaba_mat[settings.p_n:, settings.p_n:] = np.loadtxt(network_dir / f'matrix_{graph_no}.csv', delimiter=',')
    np.fill_diagonal(fgaba_mat, 0.0)

    sgaba_mat = np.zeros((settings.n_n, settings.n_n))
    sgaba_mat[:settings.p_n, settings.p_n:] = lnpn
    np.fill_diagonal(sgaba_mat, 0.0)

    return ach_mat, fgaba_mat, sgaba_mat


def build_current_input(graph_no, odor_seed, trial_seed, settings, deterministic_staging=False):
    times = build_timepoints(settings)
    active_steps = times.shape[0] - int(2 * settings.buffer / settings.sim_res)
    current_input = np.ones((settings.n_n, active_steps))

    switch_rng = np.random.RandomState(graph_no + odor_seed + trial_seed)
    if settings.switch_prob == 0.0:
        switch_state = [1]
    else:
        switch_state = [0]
    for value in switch_rng.choice(
        [0, 1],
        p=[1 - settings.switch_prob, settings.switch_prob],
        size=int(settings.blocktime / settings.min_block) - 1,
    ):
        if value == 1:
            switch_state.append(1 - switch_state[-1])
        else:
            switch_state.append(switch_state[-1])
    ts = np.repeat(switch_state, int(settings.min_block / settings.sim_res))

    pn_rng = np.random.RandomState(odor_seed)
    set_pn = np.concatenate([np.ones(9), np.zeros(81)])
    pn_rng.shuffle(set_pn)

    current_input[:settings.p_n, :] = 0.24 * (current_input[:settings.p_n, :].T * set_pn).T * ts
    current_input[settings.p_n:, :] = 0.0735 * current_input[settings.p_n:, :] * ts
    current_input = np.concatenate(
        [
            np.zeros((current_input.shape[0], int(settings.buffer / settings.sim_res))),
            current_input,
            np.zeros((current_input.shape[0], int(settings.buffer / settings.sim_res))),
        ],
        axis=1,
    )

    noise_rng = make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset=17)
    current_input += 0.05 * current_input * noise_rng.normal(size=current_input.shape)
    current_input += 0.001 * noise_rng.normal(size=current_input.shape)
    return current_input


def build_state_vector(graph_no, odor_seed, trial_seed, settings, n_syn_ach, n_syn_fgaba, n_syn_sgaba, deterministic_staging=False):
    sim_time = settings.blocktime + 2 * settings.buffer
    state_vector = np.array(
        [-45] * settings.p_n
        + [-45] * settings.l_n
        + [0.5] * (settings.n_n + 4 * settings.p_n + 3 * settings.l_n)
        + [2.4e-4] * settings.l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * settings.n_n,
        dtype=np.float64,
    )
    noise_rng = make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset=31)
    return state_vector + 0.005 * state_vector * noise_rng.normal(size=state_vector.shape)


def build_timepoints(settings):
    sim_time = settings.blocktime + 2 * settings.buffer
    return np.arange(0, sim_time, settings.sim_res)


def split_timepoints(times, settings):
    n_batch = max(1, int(math.ceil((times[-1] + settings.sim_res) / settings.batch_ms)))
    time_batches = list(np.array_split(times, n_batch))
    for index in range(1, len(time_batches)):
        time_batches[index] = np.append(time_batches[index - 1][-1], time_batches[index])
    return time_batches


def prepare_case_directory(cache_root, graph_no, odor_seed, trial_seed):
    case_dir = cache_root / f'{graph_no}_{odor_seed}_{trial_seed}'
    if case_dir.exists():
        import shutil

        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    return case_dir


def write_case_inputs(case_dir, case):
    np.save(case_dir / 'state_vector.npy', case['state_vector'])
    np.save(case_dir / 'ach_mat.npy', case['ach_mat'])
    np.save(case_dir / 'fgaba_mat.npy', case['fgaba_mat'])
    np.save(case_dir / 'sgaba_mat.npy', case['sgaba_mat'])
    np.save(case_dir / 'current_input.npy', case['current_input'])


def load_output_dataset(case_dir):
    dataset = []
    for output_file in sorted_output_files(case_dir):
        dataset.append(np.load(output_file))
    return np.concatenate(dataset)


def sorted_output_files(case_dir):
    files = [path for path in case_dir.iterdir() if path.name.startswith('output_')]
    files.sort(key=lambda path: [int(chunk) if chunk.isdigit() else chunk for chunk in re.findall(r'[^0-9]|[0-9]+', path.name)])
    return files


def make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset):
    if deterministic_staging:
        return np.random.RandomState(stable_seed(graph_no, odor_seed, trial_seed, offset))
    return np.random.RandomState()


def stable_seed(*parts):
    seed = 0
    for index, part in enumerate(parts, start=1):
        seed = (seed * 1000003 + int(part) + 97 * index) % (2**32 - 1)
    return seed


def _append_env_flags(env, key, extra_flags):
    extra_flags = str(extra_flags or '').strip()
    if not extra_flags:
        return

    existing = str(env.get(key, '') or '').strip()
    env[key] = f'{existing} {extra_flags}'.strip() if existing else extra_flags


def configure_runtime_environment(root, env=None):
    env = dict(env or {})
    backend = env.get('IODOR_BACKEND', 'tensorflow').strip().lower()
    if backend != 'jax':
        return env

    cache_dir = env.get('IODOR_JAX_COMPILATION_CACHE_DIR') or env.get('JAX_COMPILATION_CACHE_DIR')
    if not cache_dir:
        cache_dir = str(Path(root) / '__jaxcache__')
        env['IODOR_JAX_COMPILATION_CACHE_DIR'] = cache_dir

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    env.setdefault('IODOR_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
    env.setdefault('IODOR_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '-1')

    if 'IODOR_JAX_PREALLOCATE' in env:
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(env['IODOR_JAX_PREALLOCATE'])
    if 'IODOR_JAX_MEM_FRACTION' in env:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(env['IODOR_JAX_MEM_FRACTION'])
    if 'IODOR_JAX_ALLOCATOR' in env:
        env['XLA_PYTHON_CLIENT_ALLOCATOR'] = str(env['IODOR_JAX_ALLOCATOR'])

    _append_env_flags(env, 'XLA_FLAGS', env.get('IODOR_XLA_FLAGS_APPEND'))
    return env