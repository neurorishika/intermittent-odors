import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.runtime import compile_experiment, get_backend_name
from slurm.builders import (build_trial_case, load_trial_settings,
                            trial_case_to_experiment_spec)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit('Usage: single_pert_trial.py <graph_no> <odor_seed> <trial_seed>')

    graph_no  = int(argv[0])
    odor_seed = int(argv[1])
    trial_seed = int(argv[2])

    settings = load_trial_settings()
    network_dir = Path(__file__).resolve().parent.parent / 'modules' / 'networks'

    case = build_trial_case(
        graph_no, odor_seed, trial_seed,
        network_dir=network_dir,
        settings=settings,
    )
    spec = trial_case_to_experiment_spec(case, settings)

    backend = get_backend_name()
    runner = compile_experiment(spec, backend=backend)
    print(f'Using {backend} backend...')

    t_start = time.time()
    dataset, _ = runner.run_time_batches(
        case.state_vector,
        case.current_input,
        progress=tqdm,
    )
    print('Completed. Total Execution Time:', np.round(time.time() - t_start, 3), 'secs')

    Path('Data').mkdir(exist_ok=True)
    np.save(f'Data/data_{graph_no}_{odor_seed}_{trial_seed}', dataset)


if __name__ == '__main__':
    main()

