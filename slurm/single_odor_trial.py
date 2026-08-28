import os
import shutil
import sys
from pathlib import Path

from builders import (build_trial_case, configure_runtime_environment,
                      load_trial_settings, prepare_case_directory,
                      trial_case_to_experiment_spec, write_case_inputs)
from simulation import split_pnlnnetwork_timepoints
from tqdm import tqdm

from intermittent_odors.runtime import compile_experiment


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
    experiment_spec = trial_case_to_experiment_spec(
        case,
        settings,
        metadata={'graph_no': graph_no, 'odor_seed': odor_seed, 'trial_seed': trial_seed},
    )
    runner = compile_experiment(experiment_spec)

    success = False
    try:
        expanded_time_batches = []
        for time_batch in case.time_batches:
            expanded_time_batches.extend(split_pnlnnetwork_timepoints(time_batch, settings.sim_res))

        dataset, final_state = runner.run_time_batches(
            case.state_vector,
            case.current_input,
            time_batches=expanded_time_batches,
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
