import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from builders import (build_fig2_experiment_spec, load_time_batch,
                      piecewise_profile)

from intermittent_odors.runtime import compile_experiment, get_backend_name

metadata = np.load('__simcache__/metadata.npy', allow_pickle=True).item()
t = load_time_batch('__simcache__/time.npy', sys.argv[1])
current_input = np.load('__simcache__/current_input.npy')
state_vector = np.load('__simcache__/state_vector.npy')

experiment_spec = build_fig2_experiment_spec(
    metadata,
    current_input,
    state_vector,
    t,
    fgaba_mat=metadata['fgaba_mat'],
    g_fgaba=piecewise_profile(metadata['p_n'], metadata['l_n'], 0.0, 30 * 0.2 / 2 * metadata['g_gaba']),
    metadata_overrides={'script': 'simple5x3'},
)

backend = get_backend_name()
runner = compile_experiment(experiment_spec, backend=backend)
print(f'Using {backend} backend...')

t_ = time.time()
state, final_state_vector = runner.run_time_batches(
    state_vector, current_input, progress=tqdm,
)
print('Completed. Total Execution Time:', np.round(time.time() - t_, 3), 'secs')

np.save('__simcache__/state_vector.npy', final_state_vector)
np.save(f'__simoutput__/state_{sys.argv[1]}.npy', state)
