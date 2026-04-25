import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.backend import get_backend_name

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from simulation import (build_fire_thresholds, build_slurm_config,
                        sample_stride_from_sim_res, simulate_time_batches,
                        split_pnlnnetwork_timepoints)

SIM_RES = 0.01
N_N = 120
P_N = 90
L_N = 30


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 4:
        raise SystemExit('Usage: pnlnnetwork.py <graph_no> <odor_seed> <trial_seed> <batch_index>')

    graph_no, odor_seed, trial_seed, batch_index = argv
    case_dir = f"__simcache__/{graph_no}_{odor_seed}_{trial_seed}"

    ach_mat = np.load(f"{case_dir}/ach_mat.npy")
    fgaba_mat = np.load(f"{case_dir}/fgaba_mat.npy")
    sgaba_mat = np.load(f"{case_dir}/sgaba_mat.npy")
    current_input = np.load(f"{case_dir}/current_input.npy")
    state_vector = np.load(f"{case_dir}/state_vector.npy")
    timepoints = np.load(f"{case_dir}/timepoint.npy")

    config = build_slurm_config(ach_mat, fgaba_mat, sgaba_mat, n_n=N_N, p_n=P_N, l_n=L_N)
    thresholds = build_fire_thresholds(P_N, L_N)
    backend = get_backend_name()
    print(f"Using {backend} backend...")

    t_start = time.time()
    state, final_state = simulate_time_batches(
        config,
        current_input,
        state_vector,
        split_pnlnnetwork_timepoints(timepoints, SIM_RES),
        thresholds,
        backend=backend,
        sample_stride=sample_stride_from_sim_res(SIM_RES),
        sample_neurons=N_N,
        progress=tqdm,
    )
    print("Completed. Total Execution Time:", np.round(time.time() - t_start, 3), "secs")

    np.save(f"{case_dir}/output_{batch_index}.npy", state)
    np.save(f"{case_dir}/state_vector.npy", final_state)


if __name__ == '__main__':
    main()
