import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.experiment import build_experiment_spec


def normalize_by_indegree(values, matrix):
    indegree = np.sum(matrix, axis=1)
    output = np.zeros_like(values, dtype=np.float64)
    np.divide(values, indegree, out=output, where=indegree != 0)
    return output


def piecewise_profile(p_n, l_n, pn_value, ln_value):
    return np.concatenate([
        np.full(int(p_n), float(pn_value), dtype=np.float64),
        np.full(int(l_n), float(ln_value), dtype=np.float64),
    ])


BLOCKTIME_MS = 1000
BUFFER_MS = 500
SIM_RES_MS = 0.01
BASELINE_DRIVE = 0.0735


def block_pulse_filter(blocktime=BLOCKTIME_MS, sim_res=SIM_RES_MS, fraction=0.1):
    """Rectangular pulse covering the leading ``fraction`` of each block."""
    tfilter = np.zeros(int(blocktime / sim_res))
    tfilter[:int(fraction * blocktime / sim_res)] = 1
    return tfilter


def pn_ramp_filter(blocktime=BLOCKTIME_MS, sim_res=SIM_RES_MS, fraction=0.8):
    """Rise/plateau/decay envelope used for the PN perturbation drive."""
    width_red = int(fraction * blocktime / sim_res)
    return np.concatenate([
        [0, 0],
        1 - np.exp(-0.0008 * np.arange(width_red // 12)),
        0.6 + 0.4 * np.exp(-0.0002 * np.arange(7 * width_red // 12)),
        0.6 * np.exp(-0.0002 * np.arange(width_red // 3)),
        np.zeros(int(blocktime / sim_res) // 5),
    ])


def add_input_noise(current_input, scale=0.05, floor=0.001):
    """Apply the multiplicative-plus-additive drive noise.

    Draws two normal samples in that order; keep the parenthesisation, since the
    sum is formed before it is added to the drive.
    """
    return current_input + (
        scale * current_input * np.random.normal(size=current_input.shape)
        + floor * np.random.normal(size=current_input.shape)
    )


def pad_with_buffers(current_input, buffer=BUFFER_MS, sim_res=SIM_RES_MS):
    """Prepend and append silent buffer periods to a block-structured drive."""
    silence = np.zeros((current_input.shape[0], int(buffer / sim_res)))
    return np.concatenate([silence, current_input, silence], axis=1)


def block_times(n_blocks, blocktime=BLOCKTIME_MS, buffer=BUFFER_MS, sim_res=SIM_RES_MS):
    """Simulation timebase for ``n_blocks`` blocks bracketed by buffers."""
    return np.arange(0, n_blocks * blocktime + 2 * buffer, sim_res)


def build_alternating_block_pattern(samplespace, n_blocks, rest_block=None):
    """Draw a block order in which no odor is presented twice in a row.

    Consumes RNG until it lands on an order with no repeats, matching the
    rejection loop the fig2 notebook used.
    """
    order = np.random.choice(np.arange(len(samplespace)), size=n_blocks)
    while np.any(np.diff(order) == 0):
        order = np.random.choice(np.arange(len(samplespace)), size=n_blocks)

    v = [] if rest_block is None else [rest_block]
    for index in order:
        v.append(samplespace[index])
    return np.array(v)


def build_shuffled_perturbation_pattern(l_n, n_blocks, leading_zeros=1):
    """Blocks perturbing a random half of the LNs, preceded by a rest block."""
    elems = [1] * (l_n // 2) + [0] * (l_n - l_n // 2)
    v = [[0] * (l_n + leading_zeros)]
    for _ in range(n_blocks):
        np.random.shuffle(elems)
        v.append([0] * leading_zeros + elems)
    return np.array(v)


def build_block_drive_stimulus(
    n_n,
    v,
    perturbation,
    *,
    blocktime=BLOCKTIME_MS,
    buffer=BUFFER_MS,
    sim_res=SIM_RES_MS,
    baseline=BASELINE_DRIVE,
    tfilter=None,
    noise=True,
):
    """Constant baseline drive with a leading perturbation pulse per block.

    Drives the LN-only networks (fig2a,b and fig2e,f,g). ``v`` selects which
    neurons are perturbed in each block; ``perturbation`` scales that pulse.
    """
    width = int(blocktime / sim_res)
    tfilter_base = np.ones(width)
    if tfilter is None:
        tfilter = block_pulse_filter(blocktime, sim_res)

    t = block_times(len(v), blocktime, buffer, sim_res)
    current_input = np.ones((n_n, t.shape[0] - int(2 * buffer / sim_res)))
    for i in range(len(v)):
        block = slice(i * width, (i + 1) * width)
        current_input[:, block] = baseline * current_input[:, block] * tfilter_base
        current_input[:, block] += perturbation * (current_input[:, block].T * v[i]).T * tfilter

    current_input = pad_with_buffers(current_input, buffer, sim_res)
    if noise:
        current_input = add_input_noise(current_input)
    return t, current_input


def build_pn_ramp_stimulus(
    n_n,
    p_n,
    v,
    *,
    blocktime=BLOCKTIME_MS,
    buffer=BUFFER_MS,
    sim_res=SIM_RES_MS,
    baseline=BASELINE_DRIVE,
    tfilter=None,
    noise=True,
):
    """Ramped perturbation onto the PNs over a constant LN baseline (fig2c,d).

    Unlike :func:`build_block_drive_stimulus` the PN rows carry only the
    perturbation envelope, with no baseline underneath it.
    """
    width = int(blocktime / sim_res)
    tfilter_base = np.ones(width)
    if tfilter is None:
        tfilter = pn_ramp_filter(blocktime, sim_res)

    t = block_times(len(v), blocktime, buffer, sim_res)
    current_input = np.ones((n_n, t.shape[0] - int(2 * buffer / sim_res)))
    for i in range(len(v)):
        block = slice(i * width, (i + 1) * width)
        current_input[:p_n, block] = (current_input[:p_n, block].T * v[i]).T * tfilter
        current_input[p_n:, block] = baseline * current_input[p_n:, block] * tfilter_base

    current_input = pad_with_buffers(current_input, buffer, sim_res)
    if noise:
        current_input = add_input_noise(current_input)
    return t, current_input


def build_initial_state_vector(
    n_n,
    p_n,
    l_n,
    sim_time,
    *,
    n_syn_ach=0,
    n_syn_fgaba=0,
    n_syn_sgaba=0,
    jitter=0.005,
):
    """Resting state with a small multiplicative jitter per trial."""
    state_vector = np.array(
        [-45] * p_n + [-45] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4 * (10 ** (-4))] * l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * n_n
    )
    return state_vector + jitter * state_vector * np.random.normal(size=state_vector.shape)


def build_time_batches(times, n_batches, legacy_batching=False):
    """Split a rollout into chunks that overlap by one timepoint.

    Each chunk begins on the timepoint the previous chunk ended on, so the state
    carried across the seam keeps its true time label and the step across the
    seam is actually taken. The 2021 pipeline used a disjoint ``np.array_split``,
    which relabelled the carried state one step late and skipped that step
    entirely -- 27 lost steps over the 30-LN rollout. With the overlap, and with
    the sampling phase that ``intermittent_odors.runtime`` derives from each
    chunk's start time, a chunked rollout is bitwise identical to one continuous
    integration.

    Pass ``legacy_batching=True`` to restore the disjoint 2021 split.
    """
    batches = list(np.array_split(np.asarray(times, dtype=np.float64), n_batches))
    if legacy_batching:
        return batches
    for index in range(1, len(batches)):
        batches[index] = np.append(batches[index - 1][-1], batches[index])
    return batches


def save_time_batches(path, time_batches):
    """Persist time batches for the ``simple*.py`` subprocess fan-out.

    Overlapping batches differ in length by one, so they cannot be stored as a
    rectangular array the way the disjoint 2021 batches could.
    """
    store = np.empty(len(time_batches), dtype=object)
    store[:] = [np.asarray(batch, dtype=np.float64) for batch in time_batches]
    np.save(str(path), store, allow_pickle=True)


def load_time_batch(path, index):
    """Load one batch written by :func:`save_time_batches`."""
    return np.asarray(np.load(str(path), allow_pickle=True)[int(index)], dtype=np.float64)


def build_fig2_experiment_spec(
    metadata,
    current_input,
    state_vector,
    times,
    *,
    ach_mat=None,
    fgaba_mat=None,
    sgaba_mat=None,
    g_ach=None,
    g_fgaba=None,
    G_sgaba=None,
    metadata_overrides=None,
):
    n_n = int(metadata['n_n'])
    p_n = int(metadata['p_n'])
    l_n = int(metadata['l_n'])
    sim_res = float(metadata['sim_res'])

    ach_mat = _coerce_matrix(ach_mat, fallback=np.zeros((n_n, n_n), dtype=np.float64), n_n=n_n)
    fgaba_mat = _coerce_matrix(fgaba_mat, fallback=metadata['fgaba_mat'], n_n=n_n)
    sgaba_mat = _coerce_matrix(sgaba_mat, fallback=np.zeros((n_n, n_n), dtype=np.float64), n_n=n_n)

    g_ach = normalize_by_indegree(_coerce_profile(g_ach, p_n, l_n, default=(0.0, 0.0)), ach_mat)
    g_fgaba = normalize_by_indegree(_coerce_profile(g_fgaba, p_n, l_n, default=(0.0, 0.0)), fgaba_mat)
    G_sgaba = normalize_by_indegree(_coerce_profile(G_sgaba, p_n, l_n, default=(0.0, 0.0)), sgaba_mat)

    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    config = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'C_m': [1.0] * n_n,
        'g_K': [3.6] * p_n + [36.0] * l_n,
        'g_L': [0.3] * n_n,
        'E_K': [-95.0] * p_n + [-95.0] * l_n,
        'E_L': [-64.0] * p_n + [-50.0] * l_n,
        'g_Na': [7.15] * p_n,
        'g_A': [1.43] * p_n,
        'E_Na': [50.0] * p_n,
        'E_A': [-95.0] * p_n,
        'g_Ca': [5.0] * l_n,
        'g_KCa': [0.045] * l_n,
        'E_Ca': [140.0] * l_n,
        'E_KCa': [-95.0] * l_n,
        'A_Ca': 2e-4,
        'Ca0': 2.4e-4,
        't_Ca': 150.0,
        'ach_mat': ach_mat,
        'alp_ach': [10.0] * n_syn_ach,
        'bet_ach': [0.2] * n_syn_ach,
        't_max': 0.3,
        't_delay': 0.0,
        'A': [0.5] * n_n,
        'g_ach': g_ach,
        'E_ach': [0.0] * n_n,
        'fgaba_mat': fgaba_mat,
        'alp_fgaba': [10.0] * n_syn_fgaba,
        'bet_fgaba': [0.16] * n_syn_fgaba,
        'V0': [-20.0] * n_n,
        'sigma': [1.5] * n_n,
        'g_fgaba': g_fgaba,
        'E_fgaba': [-70.0] * n_n,
        'sgaba_mat': sgaba_mat,
        'K_sgaba': [100e-12] * n_syn_sgaba,
        'r1_sgaba': [1.0] * n_syn_sgaba,
        'r2_sgaba': [0.025] * n_syn_sgaba,
        'r3_sgaba': [0.1] * n_syn_sgaba,
        'r4_sgaba': [0.06] * n_syn_sgaba,
        'G_sgaba': G_sgaba,
        'E_sgaba': [-95.0] * n_n,
    }
    thresholds = [0.0] * p_n + [-20.0] * l_n
    sample_stride = max(1, int(round(1.0 / sim_res)))
    metadata_payload = {'family': 'fig2', **({} if metadata_overrides is None else metadata_overrides)}
    return build_experiment_spec(
        config,
        np.asarray(current_input, dtype=np.float64),
        np.asarray(state_vector, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        thresholds,
        input_dt=sim_res,
        sample_stride=sample_stride,
        sample_neurons=n_n,
        time_batches=(np.asarray(times, dtype=np.float64),),
        metadata=metadata_payload,
        network_metadata={'family': 'fig2'},
        stimulus_metadata={'family': 'fig2'},
    )


def _coerce_matrix(matrix, *, fallback, n_n):
    if matrix is None:
        matrix = fallback
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.ascontiguousarray(matrix.reshape((n_n, n_n)))


def _coerce_profile(profile, p_n, l_n, default):
    if profile is None:
        return piecewise_profile(p_n, l_n, default[0], default[1])

    profile = np.asarray(profile, dtype=np.float64)
    if profile.shape == (p_n + l_n,):
        return np.ascontiguousarray(profile)
    raise ValueError(f'Expected a profile of shape {(p_n + l_n,)}, got {profile.shape}')
