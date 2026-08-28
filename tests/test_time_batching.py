"""Regression tests for time-batched rollouts.

Long rollouts are integrated in chunks to bound memory. Chunking is meant to be
an implementation detail: the sampled output must not depend on how the rollout
was split. The 2021 pipeline broke that in two ways, and both are pinned here.

1. It split time with a disjoint ``np.array_split``. Each chunk's carried
   initial state was relabelled one step later than its true time, so the step
   across every seam was never taken -- 27 lost integration steps over the
   30-LN rollout. ``build_time_batches`` now overlaps chunks by one timepoint.

2. ``fig2/simple30.py`` ended each chunk with ``state[::100][:-1]``, dropping a
   real (non-duplicate) sample per chunk. The committed ``data/30LN/`` arrays
   are therefore 28 rows short and carry a 2 ms gap at each of the 27 seams,
   while ``onlyLNs.py`` plots the row index as milliseconds.

The headline assertion is that a chunked rollout is now *bit-for-bit* identical
to one continuous integration. ``--legacy-batching`` reproduces the old
behaviour so the committed dataset stays verifiable; that path is covered by the
opt-in test at the bottom.

Run::

    python -m unittest tests.test_time_batching -v

The default tests need no data files. The legacy dataset check needs
``data/30LN/`` and is gated behind ``IODOR_RUN_LEGACY_DATASET_PARITY=1``.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG2 = ROOT / 'fig2'
if str(FIG2) not in sys.path:
    sys.path.insert(0, str(FIG2))


def _has_module(name):
    return importlib.util.find_spec(name) is not None


HAS_NUMPY = _has_module('numpy')
HAS_JAX = _has_module('jax')

JAX_DEPS_MESSAGE = 'Install numpy and jax in the active interpreter to run time-batching tests.'

RUN_LEGACY_DATASET_PARITY = os.environ.get('IODOR_RUN_LEGACY_DATASET_PARITY') == '1'
LEGACY_DATASET_MESSAGE = (
    'Set IODOR_RUN_LEGACY_DATASET_PARITY=1 to replay the 30-LN rollout against the committed dataset.'
)
LEGACY_GRAPH_NO = 2
LEGACY_PERT_SEED = 59428
LEGACY_DATASET = ROOT / 'data' / '30LN' / f'LN30_data_{LEGACY_GRAPH_NO}_{LEGACY_PERT_SEED}.npy'
LEGACY_DATA_MESSAGE = f'{LEGACY_DATASET} is required; fetch it with get_data.py.'

SAMPLE_STRIDE = 10


def _build_small_case():
    """A short synthetic rollout long enough to span several sampling intervals."""
    import numpy as np

    import check_shared_tf_parity as parity

    n_n, p_n = 6, 3
    l_n = n_n - p_n
    config, _, _, _, thresholds = parity.build_case(
        seed=7, n_n=n_n, p_n=p_n, ach_density=0.3, fgaba_density=0.3, sgaba_density=0.2
    )

    times = np.arange(0.0, 4.0, 0.01, dtype=np.float64)
    sim_time = float(times[-1] + (times[1] - times[0]))
    rng = np.random.default_rng(11)
    state = parity.build_stable_state_vector(
        rng, n_n, p_n, l_n,
        int(np.sum(config['ach_mat'])),
        int(np.sum(config['fgaba_mat'])),
        int(np.sum(config['sgaba_mat'])),
        sim_time,
    )
    current_input = parity.build_stable_current_input(rng, n_n, p_n, times.shape[0])
    return config, current_input, state, times, thresholds


def _compile_small_runner():
    from intermittent_odors.experiment import build_experiment_spec
    from intermittent_odors.runtime import compile_experiment

    config, current_input, state, times, thresholds = _build_small_case()
    spec = build_experiment_spec(
        config, current_input, state, times, thresholds,
        sample_stride=SAMPLE_STRIDE,
        sample_neurons=int(config['n_n']),
    )
    return compile_experiment(spec, backend='jax'), current_input, state, times


@unittest.skipUnless(HAS_NUMPY, 'Install numpy to run time-batching helper tests.')
class SamplingGridTests(unittest.TestCase):
    """The sampling grid is global, not per batch."""

    def setUp(self):
        import numpy as np

        from builders import build_time_batches

        self.np = np
        self.times = np.arange(0.0, 100.0, 0.01, dtype=np.float64)
        self.overlapping = build_time_batches(self.times, 8)
        self.disjoint = build_time_batches(self.times, 8, legacy_batching=True)

    def test_overlapping_batches_share_their_seam(self):
        for previous, following in zip(self.overlapping, self.overlapping[1:]):
            self.assertEqual(previous[-1], following[0])

    def test_disjoint_batches_skip_a_step_at_every_seam(self):
        for previous, following in zip(self.disjoint, self.disjoint[1:]):
            self.assertNotEqual(previous[-1], following[0])

    def test_phase_tracks_the_global_grid(self):
        from intermittent_odors.runtime import sample_phase_for_batch

        self.assertEqual(sample_phase_for_batch(self.overlapping[0], SAMPLE_STRIDE), 0)
        for batch in self.overlapping[1:]:
            start_index = int(round(float(batch[0]) / 0.01))
            self.assertEqual(
                sample_phase_for_batch(batch, SAMPLE_STRIDE),
                (-start_index) % SAMPLE_STRIDE,
            )

    def test_batching_does_not_change_the_sample_count(self):
        from intermittent_odors.runtime import sampled_length_for_batch

        unbatched = sampled_length_for_batch(self.times, SAMPLE_STRIDE)
        batched = sum(sampled_length_for_batch(b, SAMPLE_STRIDE) for b in self.overlapping)
        self.assertEqual(unbatched, batched)

    def test_a_batch_shorter_than_the_grid_gap_contributes_nothing(self):
        short = self.np.arange(0.05, 0.08, 0.01, dtype=self.np.float64)
        from intermittent_odors.runtime import sampled_length_for_batch

        self.assertEqual(sampled_length_for_batch(short, SAMPLE_STRIDE), 0)


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
class ChunkedRolloutTests(unittest.TestCase):

    def test_chunked_rollout_matches_one_continuous_integration(self):
        """The headline invariant: chunking must not change a single bit."""
        import numpy as np

        from builders import build_time_batches

        runner, current_input, state, times = _compile_small_runner()
        expected, expected_final = runner.run_time_batches(
            state, current_input, time_batches=[times]
        )

        for n_batches in (2, 5, 9):
            with self.subTest(n_batches=n_batches):
                actual, actual_final = runner.run_time_batches(
                    state, current_input,
                    time_batches=build_time_batches(times, n_batches),
                )
                self.assertEqual(expected.shape, actual.shape)
                self.assertEqual(expected.tobytes(), actual.tobytes())
                self.assertEqual(expected_final.tobytes(), actual_final.tobytes())

    def test_legacy_disjoint_batching_perturbs_the_rollout(self):
        """Documents the bug the overlap fixes; the old split loses a step per seam."""
        import numpy as np

        from builders import build_time_batches

        runner, current_input, state, times = _compile_small_runner()
        expected, _ = runner.run_time_batches(state, current_input, time_batches=[times])
        legacy, _ = runner.run_time_batches(
            state, current_input,
            time_batches=build_time_batches(times, 5, legacy_batching=True),
        )
        self.assertEqual(expected.shape, legacy.shape)
        self.assertFalse(np.array_equal(expected, legacy))


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
@unittest.skipUnless(RUN_LEGACY_DATASET_PARITY, LEGACY_DATASET_MESSAGE)
@unittest.skipUnless(LEGACY_DATASET.exists(), LEGACY_DATA_MESSAGE)
class LegacyDatasetTests(unittest.TestCase):
    """``--legacy-batching`` still reproduces the committed 2021 dataset.

    This is the check that keeps ``data/30LN/`` verifiable after the batching
    fix. It replays a full five-repetition 7000 ms rollout, so it is opt-in.
    """

    def test_legacy_batching_reproduces_committed_dataset(self):
        import matplotlib
        matplotlib.use('Agg')

        import numpy as np

        import onlyLNs
        from builders import (BLOCKTIME_MS, BUFFER_MS, build_block_drive_stimulus,
                              build_shuffled_perturbation_pattern)

        metadata = onlyLNs.build_metadata(LEGACY_GRAPH_NO)
        np.random.seed(LEGACY_PERT_SEED)
        pattern = build_shuffled_perturbation_pattern(metadata['l_n'], onlyLNs.N_BLOCKS)
        sim_time = len(pattern) * BLOCKTIME_MS + 2 * BUFFER_MS
        times, current_input = build_block_drive_stimulus(
            metadata['n_n'], pattern, perturbation=0.5
        )
        state_vectors = onlyLNs.build_state_vectors(metadata, sim_time)

        regenerated = np.array(onlyLNs.simulate(
            metadata, current_input, times, state_vectors, LEGACY_PERT_SEED,
            legacy_batching=True,
        ))
        committed = np.load(LEGACY_DATASET, allow_pickle=True)

        self.assertEqual(committed.shape, regenerated.shape)
        self.assertLess(float(np.max(np.abs(regenerated - committed))), 1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
