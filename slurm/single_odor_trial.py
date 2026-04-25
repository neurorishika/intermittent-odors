import os
import shutil
import sys
from pathlib import Path

from simulation import (build_fire_thresholds, build_slurm_config,
                        sample_stride_from_sim_res, simulate_time_batches,
                        split_pnlnnetwork_timepoints)
from tqdm import tqdm
from trial_setup import (build_trial_case, configure_runtime_environment,
                         load_trial_settings, prepare_case_directory,
                         write_case_inputs)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit('Usage: single_odor_trial.py <graph_no> <odor_seed> <trial_seed>')

    graph_no = int(argv[0])
    odor_seed = int(argv[1])
    trial_seed = int(argv[2])

    root = Path(__file__).resolve().parent
    cache_root = root / '__simcache__'
    data_dir = root / 'Data'
    network_dir = root.parent / 'modules' / 'networks'

    case_dir = prepare_case_directory(cache_root, graph_no, odor_seed, trial_seed)
    settings = load_trial_settings(os.environ)
    case = build_trial_case(
        graph_no,
        odor_seed,
        trial_seed,
        network_dir,
        settings,
        deterministic_staging=os.environ.get('IODOR_DETERMINISTIC_STAGING') == '1',
    )
    write_case_inputs(case_dir, case)
    os.environ.update(configure_runtime_environment(root, os.environ.copy()))

    config = build_slurm_config(
        case['ach_mat'],
        case['fgaba_mat'],
        case['sgaba_mat'],
        n_n=settings.n_n,
        p_n=settings.p_n,
        l_n=settings.l_n,
    )
    thresholds = build_fire_thresholds(settings.p_n, settings.l_n)

    success = False
    try:
        expanded_time_batches = []
        for time_batch in case['time_batches']:
            expanded_time_batches.extend(split_pnlnnetwork_timepoints(time_batch, settings.sim_res))

        dataset, final_state = simulate_time_batches(
            config,
            case['current_input'],
            case['state_vector'],
            expanded_time_batches,
            thresholds,
            sample_stride=sample_stride_from_sim_res(settings.sim_res),
            sample_neurons=settings.n_n,
            progress=tqdm,
        )
        np.save(data_dir / f'data_{graph_no}_{odor_seed}_{trial_seed}', dataset)
        np.save(case_dir / 'state_vector.npy', final_state)
        success = True
    finally:
        if success and case_dir.exists():
            shutil.rmtree(case_dir)


if __name__ == '__main__':
    import numpy as np

    main()
