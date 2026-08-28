#!/usr/bin/env python
"""Regenerate the 30-LN perturbation dataset for one (graph, perturbation) seed pair.

This is the sweep counterpart of ``fig2.ipynb`` cell 15, which covers only
``graphno=2, pertseed=59428``. ``runSimMatrix.py`` fans this script out over the
five perturbation seeds for a given graph; together they produce the
``data/30LN/`` dataset that ``fig4/supplementary_video1.ipynb`` and
``tests/test_equivalence.py`` read.

Usage::

    python onlyLNs.py <graphno> <pertseed> [--force]

Existing outputs are left alone unless ``--force`` is passed, so a re-run cannot
silently replace the committed dataset.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import block_diag

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from builders import (BLOCKTIME_MS, BUFFER_MS, build_block_drive_stimulus,
                      build_fig2_experiment_spec, build_initial_state_vector,
                      build_shuffled_perturbation_pattern, build_time_batches,
                      piecewise_profile)

from intermittent_odors.runtime import (compile_experiment, get_backend_name,
                                        sampled_length_for_batch)

DATA_DIR = ROOT / 'data' / '30LN'
FIGURE_DIR = HERE / 'Figures'
NETWORK_DIR = ROOT / 'modules' / 'networks'

N_REPS = 5
N_BLOCKS = 5
P_N = 1
L_N = 30
N_N = P_N + L_N
G_GABA = 1.5
SIM_RES = 0.01
# Time chunks per block, matching the batch granularity the subprocess fan-out used.
BATCHES_PER_BLOCK = 4
N_TIME_BATCHES = BATCHES_PER_BLOCK * (N_BLOCKS + 2)
SAMPLE_STRIDE = max(1, int(round(1.0 / SIM_RES)))


def drop_legacy_boundary_samples(sampled, time_batches):
    """Reproduce the 2021 ``state[::100][:-1]`` truncation in ``simple30.py``.

    That slice dropped the last sample of every batch. The dropped rows were not
    duplicates, so the committed dataset is 28 rows short and carries a 2 ms gap
    at each of the 27 internal seams while downstream code reads the row index as
    milliseconds. Kept only so the committed files stay reproducible.
    """
    keep = []
    offset = 0
    for batch in time_batches:
        length = sampled_length_for_batch(batch, SAMPLE_STRIDE)
        keep.extend(range(offset, offset + length - 1))
        offset += length
    return sampled[keep]


def build_metadata(graphno):
    """Single-PN + 30-LN network with the fast-GABA graph loaded from disk."""
    fgaba_mat = block_diag(np.array([[0]]), np.load(NETWORK_DIR / f'matrix_{graphno}.npy'))
    np.fill_diagonal(fgaba_mat, 0)
    return {
        'n_n': N_N,
        'p_n': P_N,
        'l_n': L_N,
        'fgaba_mat': fgaba_mat,
        'g_gaba': G_GABA,
        'sim_res': SIM_RES,
    }


def plot_network(metadata, graphno):
    np.random.seed(783385)
    fig = plt.figure(figsize=(6, 6))
    inv_G = nx.from_numpy_array(1 - metadata['fgaba_mat'][1:, 1:], create_using=nx.Graph)
    G = nx.from_numpy_array(metadata['fgaba_mat'][1:, 1:], create_using=nx.Graph)
    pos = nx.layout.fruchterman_reingold_layout(inv_G)

    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=plt.cm.inferno(np.linspace(0.2, 0.8, 30)))
    nx.draw_networkx_edges(G, pos, node_size=200, arrowstyle='-|>', arrowsize=10,
                           width=0.5, connectionstyle='arc3, rad=0.1', edge_color='indianred')

    plt.gca().set_axis_off()
    plt.savefig(FIGURE_DIR / f'LN_only_graph_{graphno}.svg')
    plt.close(fig)


def build_state_vectors(metadata, sim_time):
    """Draw one jittered resting state per repetition, in a single RNG run."""
    return [
        build_initial_state_vector(
            metadata['n_n'], metadata['p_n'], metadata['l_n'], sim_time,
            n_syn_fgaba=int(metadata['fgaba_mat'].sum()),
        )
        for _ in range(N_REPS)
    ]


def simulate(metadata, current_input, t, state_vectors, pertseed, legacy_batching=False):
    """Run each repetition through one compiled runner, chained across time batches."""
    spec = build_fig2_experiment_spec(
        metadata,
        current_input,
        state_vectors[0],
        t,
        fgaba_mat=metadata['fgaba_mat'],
        g_fgaba=piecewise_profile(metadata['p_n'], metadata['l_n'], 0.0,
                                  30 * 0.2 / 2 * metadata['g_gaba']),
        metadata_overrides={'script': 'onlyLNs', 'pertseed': pertseed},
    )

    backend = get_backend_name()
    print(f'Using {backend} backend...')
    runner = compile_experiment(spec, backend=backend)
    time_batches = build_time_batches(t, N_TIME_BATCHES, legacy_batching=legacy_batching)

    datasets = []
    for rep, state_vector in enumerate(state_vectors):
        print(f'repetition {rep + 1}/{len(state_vectors)}')
        sampled, _ = runner.run_time_batches(
            state_vector, current_input, time_batches=time_batches, progress=tqdm,
        )
        if legacy_batching:
            sampled = drop_legacy_boundary_samples(sampled, time_batches)
        datasets.append(sampled[:, 1:31])
    return datasets


def spike_events(datasets):
    """Upward threshold crossings of -20 mV, per repetition and per LN."""
    events = []
    for dataset in datasets:
        fire = np.logical_and(dataset[:-1, :] < -20, dataset[1:, :] > -20)
        events.append(np.array(
            [np.arange(dataset.shape[0])[:-1][fire[:, i]] for i in range(fire.shape[1])],
            dtype=object,
        ))
    return np.array(events, dtype=object)


def plot_results(events, current_input, graphno, pertseed):
    fig = plt.figure(figsize=(12, 8))
    plt.eventplot(events.T.flatten(),
                  colors=np.tile(plt.cm.inferno(np.linspace(0.2, 0.8, 30)), 5).reshape(-1, 4),
                  linelengths=0.6)
    for i in range(1500, 6500, 1000):
        plt.fill_betweenx([0, 150], [i, i], [i + 100, i + 100], color='lightgray')
    plt.box(False)
    plt.xlim(0, 7000)
    plt.yticks([])
    plt.ylabel('LN Spike Raster')
    plt.xlabel('Time (in ms)')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f'LN_only_spiketrains_{graphno}_{pertseed}.svg')
    plt.close(fig)

    fig = plt.figure(figsize=(3, 8))
    for i in range(30):
        plt.plot(0.14 * i + current_input[i, :], color=plt.cm.inferno(0.2 + 0.6 * (i / 30)))
    plt.box(False)
    plt.yticks([])
    plt.ylabel('Excitatory Drive (E)')
    plt.xlabel('Time (in ms)')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f'LN_only_current_{graphno}_{pertseed}.svg')
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('graphno', type=int)
    parser.add_argument('pertseed', type=int)
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite an existing dataset.')
    parser.add_argument('--legacy-batching', action='store_true',
                        help='Reproduce the 2021 dataset bug-for-bug: disjoint time '
                             'batches and a dropped sample at each batch boundary, '
                             'giving 6972 rows instead of 7000.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    graphno, pertseed = args.graphno, args.pertseed

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f'LN30_data_{graphno}_{pertseed}.npy'

    metadata = build_metadata(graphno)
    plot_network(metadata, graphno)

    np.random.seed(pertseed)
    v = build_shuffled_perturbation_pattern(metadata['l_n'], N_BLOCKS)
    sim_time = len(v) * BLOCKTIME_MS + 2 * BUFFER_MS
    t, current_input = build_block_drive_stimulus(metadata['n_n'], v, perturbation=0.5)
    state_vectors = build_state_vectors(metadata, sim_time)

    if data_path.exists() and not args.force:
        print(f'{data_path} exists; reusing it (pass --force to recompute).')
        datasets = np.load(data_path, allow_pickle=True)
        events = np.load(DATA_DIR / f'LN30_events_{graphno}_{pertseed}.npy', allow_pickle=True)
        current_input = np.load(DATA_DIR / f'LN30_current_{graphno}_{pertseed}.npy', allow_pickle=True)
    else:
        datasets = simulate(metadata, current_input, t, state_vectors, pertseed,
                            legacy_batching=args.legacy_batching)
        events = spike_events(datasets)
        np.save(data_path, datasets, allow_pickle=True)
        np.save(DATA_DIR / f'LN30_current_{graphno}_{pertseed}.npy', current_input[:, ::100], allow_pickle=True)
        np.save(DATA_DIR / f'LN30_events_{graphno}_{pertseed}.npy', events, allow_pickle=True)
        current_input = current_input[:, ::100]

    plot_results(events, current_input, graphno, pertseed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
