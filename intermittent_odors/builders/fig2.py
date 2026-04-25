import numpy as np

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
    raise ValueError(f'Expected a profile of shape {(p_n + l_n,)}, got {profile.shape}.')