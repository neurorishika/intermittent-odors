import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.experiment import (ensure_prepared_experiment,
                                           infer_input_dt_from_batches)
from intermittent_odors.runtime import compile_experiment


def build_fire_thresholds(p_n, l_n):
    return [0.0] * p_n + [-20.0] * l_n


def build_slurm_config(ach_mat, fgaba_mat, sgaba_mat, n_n=120, p_n=90, l_n=30):
    from intermittent_odors.model import pnln_network
    from intermittent_odors.stimulus.odor import IntermittentOdorParams

    params = IntermittentOdorParams(n_n=n_n, p_n=p_n, l_n=l_n)
    g_ach, g_fgaba, G_sgaba = params.base_conductances()
    net = pnln_network(
        p_n, l_n, ach_mat, fgaba_mat, sgaba_mat,
        g_ach=g_ach, g_fgaba=g_fgaba, G_sgaba=G_sgaba,
        normalize_conductances=True,
    )
    return net.to_config_dict()


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
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_batches(time_batches),
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
        time_batches=time_batches,
    )
    runner = compile_experiment(experiment, backend=backend)
    return runner.run_time_batches(state_vector, np.asarray(current_input, dtype=np.float64), progress=progress)


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
    experiment = ensure_prepared_experiment(
        config,
        thresholds=thresholds,
        input_dt=infer_input_dt_from_batches(time_batches),
        sample_stride=sample_stride,
        sample_neurons=sample_neurons,
        time_batches=time_batches,
    )
    runner = compile_experiment(experiment, backend=backend)
    return runner.run_time_batches_batch(state_vectors, np.asarray(current_inputs, dtype=np.float64), progress=progress)