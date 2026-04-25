import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.backend import (get_backend_name, get_sampled_integrator_runner,
                            get_sampled_integrator_runner_batch)


def build_fire_thresholds(p_n, l_n):
    return [0.0] * p_n + [-20.0] * l_n


def normalize_by_indegree(values, matrix):
    indegree = np.sum(matrix, axis=1)
    output = np.zeros_like(values, dtype=np.float64)
    np.divide(values, indegree, out=output, where=indegree != 0)
    return output


def build_slurm_config(ach_mat, fgaba_mat, sgaba_mat, n_n=120, p_n=90, l_n=30):
    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    g_ach = normalize_by_indegree(
        np.array([0.0] * p_n + [2 * 90 * 0.5 * 0.1 * 0.05] * l_n, dtype=np.float64),
        ach_mat,
    )
    g_fgaba = normalize_by_indegree(
        np.array([0.3 * 6 * 1.2] * p_n + [30 * 0.2 / 2 * 1.2] * l_n, dtype=np.float64),
        fgaba_mat,
    )
    G_sgaba = normalize_by_indegree(
        np.array([0.3 * 6 * 0.03] * p_n + [0.0] * l_n, dtype=np.float64),
        sgaba_mat,
    )

    return {
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


def sample_stride_from_sim_res(sim_res):
    return max(1, int(round(1.0 / float(sim_res))))


def split_pnlnnetwork_timepoints(timepoints, sim_res):
    batches = []
    for batch_index, batch in enumerate(np.array_split(np.asarray(timepoints, dtype=np.float64), 4)):
        if batch_index > 0:
            batch = np.append(batch[0] - sim_res, batch)
        batches.append(batch)
    return batches


def simulate_time_batches(
    config,
    current_input,
    state_vector,
    time_batches,
    thresholds,
    backend=None,
    sample_stride=100,
    sample_neurons=None,
    progress=None,
):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    current_input = np.asarray(current_input, dtype=np.float64)
    runner = get_sampled_integrator_runner(
        config,
        current_input,
        thresholds,
        sample_stride,
        sample_neurons=sample_neurons,
        backend=backend_name,
    )

    if backend_name == 'jax':
        import jax
        import jax.numpy as jnp

        from shared.jax_precision import NP_DTYPE

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

    for _, time_batch in iterator:
        sampled, state_vector = runner(state_vector, time_batch)
        sampled_outputs.append(sampled)

    if backend_name == 'jax':
        if sampled_outputs:
            output = np.asarray(jnp.concatenate(sampled_outputs, axis=0), dtype=np.float64)
        else:
            output = np.zeros((0, sample_neurons), dtype=np.float64)
        final_state = np.asarray(state_vector, dtype=np.float64)
        return output, final_state

    if sampled_outputs:
        output = np.concatenate(sampled_outputs, axis=0)
    else:
        output = np.zeros((0, sample_neurons), dtype=np.float64)
    return output, state_vector


def simulate_time_batches_batch(
    config,
    current_inputs,
    state_vectors,
    time_batches,
    thresholds,
    backend=None,
    sample_stride=100,
    sample_neurons=None,
    progress=None,
):
    backend_name = get_backend_name(backend)
    sample_neurons = int(sample_neurons or config['n_n'])
    batch_size = int(np.asarray(state_vectors).shape[0])
    current_inputs = np.asarray(current_inputs, dtype=np.float64)
    runner = get_sampled_integrator_runner_batch(
        config,
        current_inputs,
        thresholds,
        sample_stride,
        sample_neurons=sample_neurons,
        backend=backend_name,
    )

    if backend_name == 'jax':
        import jax
        import jax.numpy as jnp

        from shared.jax_precision import NP_DTYPE

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

    for _, time_batch in iterator:
        sampled, state_vectors = runner(state_vectors, time_batch)
        sampled_outputs.append(sampled)

    if backend_name == 'jax':
        if sampled_outputs:
            output = np.asarray(jnp.concatenate(sampled_outputs, axis=1), dtype=np.float64)
        else:
            output = np.zeros((batch_size, 0, sample_neurons), dtype=np.float64)
        final_state = np.asarray(state_vectors, dtype=np.float64)
        return output, final_state

    if sampled_outputs:
        output = np.concatenate(sampled_outputs, axis=1)
    else:
        output = np.zeros((batch_size, 0, sample_neurons), dtype=np.float64)
    return output, state_vectors