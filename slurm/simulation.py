import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.experiment import build_experiment_spec


def sample_stride_from_sim_res(sim_res):
    return max(1, int(round(1.0 / float(sim_res))))


def split_pnlnnetwork_timepoints(timepoints, sim_res):
    batches = []
    for batch_index, batch in enumerate(np.array_split(np.asarray(timepoints, dtype=np.float64), 4)):
        if batch_index > 0:
            batch = np.append(batch[0] - sim_res, batch)
        batches.append(batch)
    return batches


def build_slurm_experiment_spec(
    ach_mat,
    fgaba_mat,
    sgaba_mat,
    current_input,
    state_vector,
    times,
    *,
    n_n=120,
    p_n=90,
    l_n=30,
    sim_res=0.01,
    time_batches=None,
    sample_neurons=None,
    metadata=None,
):
    """Assemble the standard SLURM PN/LN experiment as an ``ExperimentSpec``.

    This is the single entry point for the SLURM scripts; compile it with
    ``spec.compile(backend=...)`` and run it with the returned runner.
    """
    from intermittent_odors.model import pnln_network
    from intermittent_odors.stimulus.odor import IntermittentOdorParams

    params = IntermittentOdorParams(n_n=n_n, p_n=p_n, l_n=l_n)
    g_ach, g_fgaba, G_sgaba = params.base_conductances()
    net = pnln_network(
        p_n, l_n, ach_mat, fgaba_mat, sgaba_mat,
        g_ach=g_ach, g_fgaba=g_fgaba, G_sgaba=G_sgaba,
        normalize_conductances=True,
    )
    merged_metadata = {
        'family': 'slurm-pnlnnetwork',
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        **({} if metadata is None else metadata),
    }
    return build_experiment_spec(
        net.to_config_dict(),
        current_input,
        state_vector,
        times,
        net.fire_thresholds(),
        input_dt=sim_res,
        sample_stride=sample_stride_from_sim_res(sim_res),
        sample_neurons=n_n if sample_neurons is None else sample_neurons,
        time_batches=time_batches,
        metadata=merged_metadata,
        network_metadata={'family': 'slurm-pnlnnetwork'},
        stimulus_metadata={'family': 'slurm-pnlnnetwork'},
    )


def simulate_time_batches(spec, backend=None, progress=None):
    """Run ``spec`` batch-by-batch, returning ``(sampled_states, final_state)``."""
    runner = spec.compile(backend=backend)
    return runner.run_time_batches(spec.state_vector, spec.current_input, progress=progress)


def simulate_time_batches_batch(spec, state_vectors, current_inputs, backend=None, progress=None):
    """Run a batch of initial conditions and drives against a single ``spec``."""
    runner = spec.compile(backend=backend)
    return runner.run_time_batches_batch(
        state_vectors,
        np.asarray(current_inputs, dtype=np.float64),
        progress=progress,
    )
