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

graphno, pertseed = int(sys.argv[2]), int(sys.argv[3])

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.builders import (build_fig2_experiment_spec,
                                         piecewise_profile)
from intermittent_odors.runtime import compile_experiment, get_backend_name

metadata = np.load(f'__simcache__/metadata_{graphno}_{pertseed}.npy', allow_pickle=True).item()
sim_res = float(metadata['sim_res'])
t = np.load(f'__simcache__/time_{graphno}_{pertseed}.npy')[int(sys.argv[1])]
current_input = np.load(f'__simcache__/current_input_{graphno}_{pertseed}.npy')
state_vector = np.load(f'__simcache__/state_vector_{graphno}_{pertseed}.npy')

experiment_spec = build_fig2_experiment_spec(
    metadata,
    current_input,
    state_vector,
    t,
    fgaba_mat=metadata['fgaba_mat'],
    g_fgaba=piecewise_profile(metadata['p_n'], metadata['l_n'], 0.0, 30 * 0.2 / 2 * metadata['g_gaba']),
    metadata_overrides={'script': 'simple30'},
)

backend = get_backend_name()
runner = compile_experiment(experiment_spec, backend=backend)
print(f'Using {backend} backend...')

t_ = time.time()
states = []

for batch_index, batch_times in tqdm(enumerate(np.array_split(t, 1))):
    if batch_index > 0:
        batch_times = np.append(batch_times[0] - sim_res, batch_times)

    state = runner.run(state_vector, current_input, batch_times)
    state_vector = state[-1, :]
    states.append(state[::experiment_spec.sample_stride, :][:-1, :])

state = np.concatenate(states)
print('Completed. Total Execution Time:', np.round(time.time() - t_, 3), 'secs')

np.save(f'__simcache__/state_vector_{graphno}_{pertseed}.npy', state_vector)
np.save(f'__simoutput__/state_{sys.argv[1]}_{graphno}_{pertseed}.npy', state)
np.save(f'__simoutput__/state_{sys.argv[1]}_{graphno}_{pertseed}.npy', state)
