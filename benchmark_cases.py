from pathlib import Path

import numpy as np

from intermittent_odors.experiment import build_experiment_spec
from slurm.builders import (TrialSettings, build_timepoints, build_trial_case,
                            split_timepoints, stable_seed,
                            trial_case_to_experiment_spec)

ROOT = Path(__file__).resolve().parent


def normalize_by_indegree(values, matrix):
    indegree = np.sum(matrix, axis=1)
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values, indegree, out=out, where=indegree != 0)
    return out


def build_stable_current_input(rng, n_n, p_n, steps):
    current_input = np.zeros((n_n, steps), dtype=np.float64)
    activation = rng.choice([0.0, 1.0], size=steps, p=[0.375, 0.625])
    if not np.any(activation):
        activation[rng.integers(steps)] = 1.0

    pn_drive = np.zeros(p_n, dtype=np.float64)
    active_pn = max(1, int(np.ceil(p_n * 0.3)))
    pn_drive[rng.choice(p_n, size=active_pn, replace=False)] = 1.0

    current_input[:p_n, :] = 0.18 * pn_drive[:, None] * activation[None, :]
    if n_n > p_n:
        ln_activation = np.roll(activation, 1)
        current_input[p_n:, :] = 0.055 * ln_activation[None, :]

    current_input += 0.03 * current_input * rng.normal(size=current_input.shape)
    current_input += 1e-4 * rng.normal(size=current_input.shape)
    return np.clip(current_input, 0.0, None)


def build_stable_state_vector(rng, n_n, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time):
    state_vector = np.array(
        [-45.0] * p_n
        + [-45.0] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4e-4] * l_n
        + [0.0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1.0)] * n_n,
        dtype=np.float64,
    )

    state_vector[:n_n] += rng.normal(scale=0.75, size=n_n)

    gate_start = n_n
    gate_stop = 2 * n_n + 4 * p_n + 3 * l_n
    state_vector[gate_start:gate_stop] = np.clip(
        state_vector[gate_start:gate_stop] + 0.04 * rng.normal(size=gate_stop - gate_start),
        0.0,
        1.0,
    )

    ca_start = 2 * n_n + 4 * p_n + 3 * l_n
    ca_stop = ca_start + l_n
    if l_n:
        state_vector[ca_start:ca_stop] = np.clip(
            state_vector[ca_start:ca_stop] * (1.0 + 0.03 * rng.normal(size=l_n)),
            1e-6,
            None,
        )

    syn_start = 6 * n_n
    syn_stop = syn_start + n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba
    if syn_stop > syn_start:
        state_vector[syn_start:syn_stop] = np.clip(
            0.01 * rng.normal(size=syn_stop - syn_start),
            0.0,
            0.05,
        )

    fire_start = syn_stop
    state_vector[fire_start:] += 0.1 * rng.normal(size=n_n)
    return state_vector


def build_case(seed, n_n, p_n, ach_density, fgaba_density, sgaba_density):
    rng = np.random.default_rng(seed)
    l_n = n_n - p_n
    ach_mat = (rng.random((n_n, n_n)) < ach_density).astype(np.float64)
    fgaba_mat = (rng.random((n_n, n_n)) < fgaba_density).astype(np.float64)
    sgaba_mat = (rng.random((n_n, n_n)) < sgaba_density).astype(np.float64)
    np.fill_diagonal(ach_mat, 0.0)
    np.fill_diagonal(fgaba_mat, 0.0)
    np.fill_diagonal(sgaba_mat, 0.0)

    g_ach = np.concatenate([np.zeros(p_n), np.full(l_n, 0.225)])
    g_fgaba = np.concatenate([np.full(p_n, 2.16), np.full(l_n, 3.6)])
    G_sgaba = np.concatenate([np.full(p_n, 0.054), np.zeros(l_n)])

    g_ach = normalize_by_indegree(g_ach, ach_mat)
    g_fgaba = normalize_by_indegree(g_fgaba, fgaba_mat)
    G_sgaba = normalize_by_indegree(G_sgaba, sgaba_mat)

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
        'E_K': [-95.0] * n_n,
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

    times = np.arange(0.0, 0.08, 0.01, dtype=np.float64)
    sim_time = float(times[-1] + (times[1] - times[0]))
    state = build_stable_state_vector(rng, n_n, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time)
    current_input = build_stable_current_input(rng, n_n, p_n, times.shape[0])
    thresholds = [0.0] * p_n + [-20.0] * l_n
    return config, current_input, state, times, thresholds


def build_repo_production_case(graph_no=1, odor_seed=59428, trial_seed=1):
    n_n = 120
    p_n = 90
    l_n = 30

    pPNPN = 0.0
    pPNLN = 0.1
    pLNPN = 0.2

    ach_rng = np.random.default_rng(64163 + graph_no)
    ach_mat = np.zeros((n_n, n_n), dtype=np.float64)
    ach_mat[p_n:, :p_n] = ach_rng.choice([0.0, 1.0], size=(l_n, p_n), p=(1 - pPNLN, pPNLN))
    ach_mat[:p_n, :p_n] = ach_rng.choice([0.0, 1.0], size=(p_n, p_n), p=(1 - pPNPN, pPNPN))

    lnpn = np.zeros((p_n, l_n), dtype=np.float64)
    stride = int(p_n / l_n)
    spread = (round(pLNPN * p_n) // 2) * 2 + 1
    center = 0
    index = np.arange(p_n)
    for column in range(l_n):
        idx = index[np.arange(center - spread // 2, 1 + center + spread // 2) % p_n]
        lnpn[idx, column] = 1.0
        center += stride

    lnln = np.loadtxt(ROOT / 'modules' / 'networks' / f'matrix_{graph_no}.csv', delimiter=',')

    fgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
    fgaba_mat[:p_n, p_n:] = lnpn
    fgaba_mat[p_n:, p_n:] = lnln
    np.fill_diagonal(fgaba_mat, 0.0)

    sgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
    sgaba_mat[:p_n, p_n:] = lnpn
    np.fill_diagonal(sgaba_mat, 0.0)

    g_ach = np.concatenate([np.zeros(p_n), np.full(l_n, 2 * 90 * 0.5 * 0.1 * 0.05)])
    g_fgaba = np.concatenate([np.full(p_n, 0.3 * 6 * 1.2), np.full(l_n, 30 * 0.2 / 2 * 1.2)])
    G_sgaba = np.concatenate([np.full(p_n, 0.3 * 6 * 0.03), np.zeros(l_n)])

    g_ach = normalize_by_indegree(g_ach, ach_mat)
    g_fgaba = normalize_by_indegree(g_fgaba, fgaba_mat)
    G_sgaba = normalize_by_indegree(G_sgaba, sgaba_mat)

    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    current_rng = np.random.default_rng(graph_no + odor_seed + trial_seed)
    current_input = np.zeros((n_n, 8), dtype=np.float64)
    set_pn = np.concatenate([np.ones(9), np.zeros(81)])
    current_rng.shuffle(set_pn)
    ts = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)
    current_input[:p_n, :] = 0.24 * set_pn[:, None] * ts[None, :]
    current_input[p_n:, :] = 0.0735 * ts[None, :]
    current_input += 0.05 * current_input * current_rng.normal(size=current_input.shape) + 0.001 * current_rng.normal(size=current_input.shape)

    sim_time = 0.08
    state_vector = np.array(
        [-45] * p_n + [-45] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4 * (10 ** (-4))] * l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * n_n,
        dtype=np.float64,
    )
    state_rng = np.random.default_rng(odor_seed + trial_seed)
    state_vector = state_vector + 0.005 * state_vector * state_rng.normal(size=state_vector.shape)

    config = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'C_m': [1.0] * n_n,
        'g_K': [3.6] * p_n + [36.0] * l_n,
        'g_L': [0.3] * n_n,
        'E_K': [-95.0] * n_n,
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
    times = np.arange(0.0, sim_time, 0.01, dtype=np.float64)
    thresholds = [0.0] * p_n + [-20.0] * l_n
    return config, current_input, state_vector, times, thresholds


def infer_pn_ln_counts(total_neurons, p_neurons=None):
    if total_neurons < 2:
        raise ValueError('total_neurons must be at least 2.')

    if p_neurons is None:
        p_neurons = int(round(total_neurons * 0.75))

    p_neurons = int(p_neurons)
    l_neurons = int(total_neurons) - p_neurons
    if p_neurons < 1 or l_neurons < 1:
        raise ValueError('Both PN and LN counts must be positive.')
    return p_neurons, l_neurons


def build_realistic_trial_case(
    total_neurons=120,
    p_neurons=None,
    graph_no=1,
    odor_seed=59428,
    trial_seed=1,
    blocktime_ms=1000,
    buffer_ms=100,
    sim_res_ms=0.01,
    batch_ms=250,
    switch_prob=0.1,
    sample_every_ms=1.0,
):
    p_n, l_n = infer_pn_ln_counts(total_neurons, p_neurons)
    settings = TrialSettings(
        n_n=int(total_neurons),
        p_n=p_n,
        l_n=l_n,
        blocktime=int(blocktime_ms),
        buffer=int(buffer_ms),
        sim_res=float(sim_res_ms),
        batch_ms=int(batch_ms),
        switch_prob=float(switch_prob),
    )

    if settings.n_n == 120 and settings.p_n == 90 and settings.l_n == 30:
        case = build_trial_case(
            graph_no,
            odor_seed,
            trial_seed,
            ROOT / 'modules' / 'networks',
            settings,
            deterministic_staging=True,
        )
        ach_mat = np.asarray(case.ach_mat, dtype=np.float64)
        fgaba_mat = np.asarray(case.fgaba_mat, dtype=np.float64)
        sgaba_mat = np.asarray(case.sgaba_mat, dtype=np.float64)
        current_input = np.asarray(case.current_input, dtype=np.float64)
        state_vector = np.asarray(case.state_vector, dtype=np.float64)
        time_batches = [np.asarray(batch, dtype=np.float64) for batch in case.time_batches]
        times = np.asarray(case.times, dtype=np.float64)
        topology = 'exact-production'
        experiment_spec = trial_case_to_experiment_spec(
            case,
            settings,
            metadata={'topology': topology, 'case': 'realistic-slurm'},
        )
    else:
        ach_mat, fgaba_mat, sgaba_mat = build_scaled_connectivity_matrices(graph_no, settings)
        current_input = build_scaled_current_input(graph_no, odor_seed, trial_seed, settings)
        state_vector = build_scaled_state_vector(graph_no, odor_seed, trial_seed, settings, ach_mat, fgaba_mat, sgaba_mat)
        times = build_timepoints(settings)
        time_batches = split_timepoints(times, settings)
        topology = 'scaled-production-like'
        thresholds = [0.0] * settings.p_n + [-20.0] * settings.l_n
        experiment_spec = build_experiment_spec(
            build_realistic_config(settings, ach_mat, fgaba_mat, sgaba_mat),
            current_input,
            state_vector,
            times,
            thresholds,
            input_dt=settings.sim_res,
            sample_stride=max(1, int(round(float(sample_every_ms) / settings.sim_res))),
            sample_neurons=settings.n_n,
            time_batches=time_batches,
            metadata={'topology': topology, 'case': 'realistic-slurm'},
            network_metadata={'family': 'scaled-production-like'},
            stimulus_metadata={'family': 'scaled-production-like'},
        )

    config = build_realistic_config(settings, ach_mat, fgaba_mat, sgaba_mat)
    sample_stride = max(1, int(round(float(sample_every_ms) / settings.sim_res)))
    thresholds = [0.0] * settings.p_n + [-20.0] * settings.l_n
    return {
        'experiment_spec': experiment_spec,
        'config': config,
        'current_input': current_input,
        'state_vector': state_vector,
        'times': times,
        'time_batches': [np.asarray(batch, dtype=np.float64) for batch in time_batches],
        'thresholds': thresholds,
        'sample_stride': sample_stride,
        'sample_neurons': settings.n_n,
        'simulated_ms': float(times[-1] + settings.sim_res) if times.size else 0.0,
        'topology': topology,
    }


def build_scaled_connectivity_matrices(graph_no, settings):
    ach_mat = np.zeros((settings.n_n, settings.n_n), dtype=np.float64)
    ach_rng = np.random.default_rng(64163 + graph_no)
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
    np.fill_diagonal(ach_mat, 0.0)

    lnpn = np.zeros((settings.p_n, settings.l_n), dtype=np.float64)
    stride = max(1, int(settings.p_n / settings.l_n))
    spread = max(1, (round(settings.p_lnpn * settings.p_n) // 2) * 2 + 1)
    center = 0
    index = np.arange(settings.p_n)
    for column in range(settings.l_n):
        idx = index[np.arange(center - spread // 2, 1 + center + spread // 2) % settings.p_n]
        lnpn[idx, column] = 1.0
        center += stride

    fgaba_mat = np.zeros((settings.n_n, settings.n_n), dtype=np.float64)
    fgaba_mat[:settings.p_n, settings.p_n:] = lnpn
    fgaba_mat[settings.p_n:, settings.p_n:] = build_scaled_lnln_matrix(graph_no, settings.l_n)
    np.fill_diagonal(fgaba_mat, 0.0)

    sgaba_mat = np.zeros((settings.n_n, settings.n_n), dtype=np.float64)
    sgaba_mat[:settings.p_n, settings.p_n:] = lnpn
    np.fill_diagonal(sgaba_mat, 0.0)

    return ach_mat, fgaba_mat, sgaba_mat


def build_scaled_lnln_matrix(graph_no, l_neurons):
    if l_neurons == 30:
        return np.loadtxt(ROOT / 'modules' / 'networks' / f'matrix_{graph_no}.csv', delimiter=',').astype(np.float64)

    base = np.loadtxt(ROOT / 'modules' / 'networks' / f'matrix_{graph_no}.csv', delimiter=',').astype(np.float64)
    density = float(np.count_nonzero(base) / base.size)
    rng = np.random.default_rng(stable_seed(graph_no, l_neurons, 7919))
    lnln = (rng.random((l_neurons, l_neurons)) < density).astype(np.float64)
    np.fill_diagonal(lnln, 0.0)
    return lnln


def build_scaled_current_input(graph_no, odor_seed, trial_seed, settings):
    times = build_timepoints(settings)
    buffer_steps = int(settings.buffer / settings.sim_res)
    active_steps = times.shape[0] - 2 * buffer_steps
    current_input = np.ones((settings.n_n, active_steps), dtype=np.float64)

    switch_rng = np.random.default_rng(graph_no + odor_seed + trial_seed)
    if settings.switch_prob == 0.0:
        switch_state = [1]
    else:
        switch_state = [0]
    for value in switch_rng.choice(
        [0, 1],
        p=[1 - settings.switch_prob, settings.switch_prob],
        size=max(0, int(settings.blocktime / settings.min_block) - 1),
    ):
        if value == 1:
            switch_state.append(1 - switch_state[-1])
        else:
            switch_state.append(switch_state[-1])

    repeat_factor = max(1, int(settings.min_block / settings.sim_res))
    ts = np.repeat(switch_state, repeat_factor)[:active_steps]
    if ts.shape[0] < active_steps:
        ts = np.pad(ts, (0, active_steps - ts.shape[0]), mode='edge')

    active_pn = max(1, int(round(settings.p_n * 0.1)))
    set_pn = np.concatenate([np.ones(active_pn), np.zeros(settings.p_n - active_pn)])
    pn_rng = np.random.default_rng(odor_seed)
    pn_rng.shuffle(set_pn)

    current_input[:settings.p_n, :] = 0.24 * set_pn[:, None] * ts[None, :]
    current_input[settings.p_n:, :] = 0.0735 * ts[None, :]
    current_input = np.concatenate(
        [
            np.zeros((settings.n_n, buffer_steps), dtype=np.float64),
            current_input,
            np.zeros((settings.n_n, buffer_steps), dtype=np.float64),
        ],
        axis=1,
    )

    noise_rng = np.random.default_rng(stable_seed(graph_no, odor_seed, trial_seed, 17))
    current_input += 0.05 * current_input * noise_rng.normal(size=current_input.shape)
    current_input += 0.001 * noise_rng.normal(size=current_input.shape)
    return current_input


def build_scaled_state_vector(graph_no, odor_seed, trial_seed, settings, ach_mat, fgaba_mat, sgaba_mat):
    sim_time = settings.blocktime + 2 * settings.buffer
    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))
    state_vector = np.array(
        [-45] * settings.p_n
        + [-45] * settings.l_n
        + [0.5] * (settings.n_n + 4 * settings.p_n + 3 * settings.l_n)
        + [2.4e-4] * settings.l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * settings.n_n,
        dtype=np.float64,
    )
    noise_rng = np.random.default_rng(stable_seed(graph_no, odor_seed, trial_seed, 31))
    return state_vector + 0.005 * state_vector * noise_rng.normal(size=state_vector.shape)


def build_realistic_config(settings, ach_mat, fgaba_mat, sgaba_mat):
    expected_ln_inputs = max(1.0, settings.l_n * settings.p_lnpn)
    g_ach = np.concatenate(
        [
            np.zeros(settings.p_n, dtype=np.float64),
            np.full(settings.l_n, 2 * settings.p_n * 0.5 * settings.p_pnln * 0.05, dtype=np.float64),
        ]
    )
    g_fgaba = np.concatenate(
        [
            np.full(settings.p_n, 0.3 * expected_ln_inputs * 1.2, dtype=np.float64),
            np.full(settings.l_n, settings.l_n * settings.p_lnpn / 2 * 1.2, dtype=np.float64),
        ]
    )
    G_sgaba = np.concatenate(
        [
            np.full(settings.p_n, 0.3 * expected_ln_inputs * 0.03, dtype=np.float64),
            np.zeros(settings.l_n, dtype=np.float64),
        ]
    )

    g_ach = normalize_by_indegree(g_ach, ach_mat)
    g_fgaba = normalize_by_indegree(g_fgaba, fgaba_mat)
    G_sgaba = normalize_by_indegree(G_sgaba, sgaba_mat)

    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    return {
        'n_n': settings.n_n,
        'p_n': settings.p_n,
        'l_n': settings.l_n,
        'C_m': [1.0] * settings.n_n,
        'g_K': [3.6] * settings.p_n + [36.0] * settings.l_n,
        'g_L': [0.3] * settings.n_n,
        'E_K': [-95.0] * settings.n_n,
        'E_L': [-64.0] * settings.p_n + [-50.0] * settings.l_n,
        'g_Na': [7.15] * settings.p_n,
        'g_A': [1.43] * settings.p_n,
        'E_Na': [50.0] * settings.p_n,
        'E_A': [-95.0] * settings.p_n,
        'g_Ca': [5.0] * settings.l_n,
        'g_KCa': [0.045] * settings.l_n,
        'E_Ca': [140.0] * settings.l_n,
        'E_KCa': [-95.0] * settings.l_n,
        'A_Ca': 2e-4,
        'Ca0': 2.4e-4,
        't_Ca': 150.0,
        'ach_mat': ach_mat,
        'alp_ach': [10.0] * n_syn_ach,
        'bet_ach': [0.2] * n_syn_ach,
        't_max': 0.3,
        't_delay': 0.0,
        'A': [0.5] * settings.n_n,
        'g_ach': g_ach,
        'E_ach': [0.0] * settings.n_n,
        'fgaba_mat': fgaba_mat,
        'alp_fgaba': [10.0] * n_syn_fgaba,
        'bet_fgaba': [0.16] * n_syn_fgaba,
        'V0': [-20.0] * settings.n_n,
        'sigma': [1.5] * settings.n_n,
        'g_fgaba': g_fgaba,
        'E_fgaba': [-70.0] * settings.n_n,
        'sgaba_mat': sgaba_mat,
        'K_sgaba': [100e-12] * n_syn_sgaba,
        'r1_sgaba': [1.0] * n_syn_sgaba,
        'r2_sgaba': [0.025] * n_syn_sgaba,
        'r3_sgaba': [0.1] * n_syn_sgaba,
        'r4_sgaba': [0.06] * n_syn_sgaba,
        'G_sgaba': G_sgaba,
        'E_sgaba': [-95.0] * settings.n_n,
    }