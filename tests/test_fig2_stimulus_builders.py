"""Parity tests for the fig2 stimulus builders.

``fig2/builders.py`` grew stimulus constructors lifted out of the inline numpy
in ``fig2/fig2.ipynb`` cells 4, 9 and 15. Those cells sit behind
``recalculate = False``, so a regression in the builders would stay invisible
until somebody flips the flag and regenerates the figure data.

Each test replays the original inline cell body verbatim and asserts the
builder output is *bit-for-bit identical* — not merely close. Bitwise equality
is the right bar here: the builders must consume the global numpy RNG in the
same order and combine floats with the same associativity as the code they
replaced, otherwise regenerated data would silently diverge from what is
committed under ``data/``.

Run::

    python -m unittest tests.test_fig2_stimulus_builders -v

No backend, GPU, or data files are required.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG2 = ROOT / 'fig2'
if str(FIG2) not in sys.path:
    sys.path.insert(0, str(FIG2))

HAS_NUMPY = importlib.util.find_spec('numpy') is not None

NUMPY_MESSAGE = 'Install numpy in the active interpreter to run fig2 stimulus builder tests.'

BLOCKTIME = 1000
BUFFER = 500
SIM_RES = 0.01


def _assert_bitwise(test, expected, actual, label):
    test.assertEqual(expected.shape, actual.shape, f'{label}: shape mismatch')
    test.assertEqual(expected.dtype, actual.dtype, f'{label}: dtype mismatch')
    test.assertEqual(expected.tobytes(), actual.tobytes(), f'{label}: not bit-for-bit identical')


@unittest.skipUnless(HAS_NUMPY, NUMPY_MESSAGE)
class Fig2StimulusBuilderParityTests(unittest.TestCase):
    """Builders must reproduce the original inline notebook cells exactly."""

    def test_cell4_three_ln_block_stimulus(self):
        """fig2.ipynb cell 4 — 3LN network, rectangular perturbation pulses."""
        import numpy as np

        from builders import (build_alternating_block_pattern,
                              build_block_drive_stimulus,
                              build_initial_state_vector)

        n_n, p_n, l_n = 1 + 5 * 3, 1, 5 * 3
        n_syn_fgaba = 210
        samplespace = [[0] + [1, 0, 0] * 5, [0] + [0, 1, 0] * 5, [0] + [0, 0, 1] * 5]
        rest_block = [0] + [0, 0, 0] * 5

        # --- original inline cell body -------------------------------------
        np.random.seed(74932)
        v = [rest_block]
        order = np.random.choice(np.arange(3), size=9)
        while np.any(np.diff(order) == 0):
            order = np.random.choice(np.arange(3), size=9)
        for i in order:
            v.append(samplespace[i])
        v = np.array(v)

        width = int(BLOCKTIME / SIM_RES)
        tfilter_base = np.ones(width)
        width_red = int(0.1 * BLOCKTIME / SIM_RES)
        tfilter = np.zeros_like(tfilter_base)
        tfilter[:width_red] = 1
        sim_time = len(v) * BLOCKTIME + 2 * BUFFER
        t = np.arange(0, sim_time, SIM_RES)
        current_input = np.ones((n_n, t.shape[0] - int(2 * BUFFER / SIM_RES)))
        for i in range(len(v)):
            block = slice(i * width, (i + 1) * width)
            current_input[:, block] = 0.0735 * current_input[:, block] * tfilter_base
            current_input[:, block] += 0.15 * (current_input[:, block].T * v[i]).T * tfilter
        pad = np.zeros((current_input.shape[0], int(BUFFER / SIM_RES)))
        current_input = np.concatenate([pad, current_input, pad], axis=1)
        current_input += (0.05 * current_input * np.random.normal(size=current_input.shape)
                          + 0.001 * np.random.normal(size=current_input.shape))

        state_vector = np.array(
            [-45] * p_n + [-45] * l_n + [0.5] * (n_n + 4 * p_n + 3 * l_n)
            + [2.4 * (10 ** (-4))] * l_n + [0] * n_syn_fgaba + [-(sim_time + 1)] * n_n
        )
        state_vector = state_vector + 0.005 * state_vector * np.random.normal(size=state_vector.shape)
        # -------------------------------------------------------------------

        np.random.seed(74932)
        built_v = build_alternating_block_pattern(samplespace, 9, rest_block=rest_block)
        built_t, built_input = build_block_drive_stimulus(n_n, built_v, perturbation=0.15)
        built_state = build_initial_state_vector(
            n_n, p_n, l_n, len(built_v) * BLOCKTIME + 2 * BUFFER, n_syn_fgaba=n_syn_fgaba,
        )

        _assert_bitwise(self, v, built_v, 'cell4 v')
        _assert_bitwise(self, t, built_t, 'cell4 t')
        _assert_bitwise(self, current_input, built_input, 'cell4 current_input')
        _assert_bitwise(self, state_vector, built_state, 'cell4 state_vector')

    def test_cell9_pn_ramp_stimulus(self):
        """fig2.ipynb cell 9 — 3PN3LN network, ramped PN perturbation."""
        import numpy as np

        from builders import (build_alternating_block_pattern,
                              build_initial_state_vector, build_pn_ramp_stimulus)

        n_n, p_n, l_n = 5 * 3 + 5 * 3, 5 * 3, 5 * 3
        n_syn_ach, n_syn_fgaba, n_syn_sgaba = 15, 240, 240
        samplespace = [[0.31, 0, 0] * 5, [0, 0.31, 0] * 5, [0, 0, 0.31] * 5]

        # --- original inline cell body -------------------------------------
        np.random.seed(8204491)
        v = []
        order = np.random.choice(np.arange(3), size=10)
        while np.any(np.diff(order) == 0):
            order = np.random.choice(np.arange(3), size=10)
        for i in order:
            v.append(samplespace[i])
        v = np.array(v)

        width = int(BLOCKTIME / SIM_RES)
        tfilter_base = np.ones(width)
        width_red = int(0.8 * BLOCKTIME / SIM_RES)
        tfilter = np.concatenate([
            [0, 0],
            1 - np.exp(-0.0008 * np.arange(width_red // 12)),
            0.6 + 0.4 * np.exp(-0.0002 * np.arange(7 * width_red // 12)),
            0.6 * np.exp(-0.0002 * np.arange(width_red // 3)),
            np.zeros(int(BLOCKTIME / SIM_RES) // 5),
        ])
        sim_time = len(v) * BLOCKTIME + 2 * BUFFER
        t = np.arange(0, sim_time, SIM_RES)
        current_input = np.ones((n_n, t.shape[0] - int(2 * BUFFER / SIM_RES)))
        for i in range(len(v)):
            block = slice(i * width, (i + 1) * width)
            current_input[:p_n, block] = (current_input[:p_n, block].T * v[i]).T * tfilter
            current_input[p_n:, block] = 0.0735 * current_input[p_n:, block] * tfilter_base
        pad = np.zeros((current_input.shape[0], int(BUFFER / SIM_RES)))
        current_input = np.concatenate([pad, current_input, pad], axis=1)
        current_input += (0.05 * current_input * np.random.normal(size=current_input.shape)
                          + 0.001 * np.random.normal(size=current_input.shape))

        state_vector = np.array(
            [-45] * p_n + [-45] * l_n + [0.5] * (n_n + 4 * p_n + 3 * l_n)
            + [2.4 * (10 ** (-4))] * l_n
            + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
            + [-(sim_time + 1)] * n_n
        )
        state_vector = state_vector + 0.005 * state_vector * np.random.normal(size=state_vector.shape)
        # -------------------------------------------------------------------

        np.random.seed(8204491)
        built_v = build_alternating_block_pattern(samplespace, 10)
        built_t, built_input = build_pn_ramp_stimulus(n_n, p_n, built_v)
        built_state = build_initial_state_vector(
            n_n, p_n, l_n, len(built_v) * BLOCKTIME + 2 * BUFFER,
            n_syn_ach=n_syn_ach, n_syn_fgaba=n_syn_fgaba, n_syn_sgaba=n_syn_sgaba,
        )

        _assert_bitwise(self, v, built_v, 'cell9 v')
        _assert_bitwise(self, t, built_t, 'cell9 t')
        _assert_bitwise(self, current_input, built_input, 'cell9 current_input')
        _assert_bitwise(self, state_vector, built_state, 'cell9 state_vector')

    def test_cell15_thirty_ln_perturbation_stimulus(self):
        """fig2.ipynb cell 15 / onlyLNs.py — 30LN network, five repetitions.

        Checked across all five perturbation seeds that ``runSimMatrix.py``
        sweeps, since those drive the committed ``data/30LN`` dataset.
        """
        import numpy as np

        from builders import (build_block_drive_stimulus,
                              build_initial_state_vector,
                              build_shuffled_perturbation_pattern)

        n_n, p_n, l_n = 1 + 30, 1, 30
        n_syn_fgaba = 396
        n_reps = 5

        for pertseed in (59428, 13674, 84932, 72957, 85036):
            with self.subTest(pertseed=pertseed):
                # --- original inline cell body -----------------------------
                np.random.seed(pertseed)
                v = [[0] * 31]
                elems = [1] * 15 + [0] * 15
                np.random.shuffle(elems)
                v.append([0] + elems)
                for _ in range(4):
                    np.random.shuffle(elems)
                    v.append([0] + elems)
                v = np.array(v)

                width = int(BLOCKTIME / SIM_RES)
                tfilter_base = np.ones(width)
                width_red = int(0.1 * BLOCKTIME / SIM_RES)
                tfilter = np.zeros_like(tfilter_base)
                tfilter[:width_red] = 1
                sim_time = len(v) * BLOCKTIME + 2 * BUFFER
                t = np.arange(0, sim_time, SIM_RES)
                current_input = np.ones((n_n, t.shape[0] - int(2 * BUFFER / SIM_RES)))
                for i in range(len(v)):
                    block = slice(i * width, (i + 1) * width)
                    current_input[:, block] = 0.0735 * current_input[:, block] * tfilter_base
                    current_input[:, block] += 0.5 * (current_input[:, block].T * v[i]).T * tfilter
                pad = np.zeros((current_input.shape[0], int(BUFFER / SIM_RES)))
                current_input = np.concatenate([pad, current_input, pad], axis=1)
                current_input += (0.05 * current_input * np.random.normal(size=current_input.shape)
                                  + 0.001 * np.random.normal(size=current_input.shape))

                state_vectors = []
                for _ in range(n_reps):
                    state_vector = np.array(
                        [-45] * p_n + [-45] * l_n + [0.5] * (n_n + 4 * p_n + 3 * l_n)
                        + [2.4 * (10 ** (-4))] * l_n + [0] * n_syn_fgaba
                        + [-(sim_time + 1)] * n_n
                    )
                    state_vectors.append(
                        state_vector + 0.005 * state_vector * np.random.normal(size=state_vector.shape)
                    )
                state_vectors = np.array(state_vectors)
                # -----------------------------------------------------------

                np.random.seed(pertseed)
                built_v = build_shuffled_perturbation_pattern(30, 5)
                built_t, built_input = build_block_drive_stimulus(n_n, built_v, perturbation=0.5)
                built_sim_time = len(built_v) * BLOCKTIME + 2 * BUFFER
                built_states = np.array([
                    build_initial_state_vector(n_n, p_n, l_n, built_sim_time, n_syn_fgaba=n_syn_fgaba)
                    for _ in range(n_reps)
                ])

                _assert_bitwise(self, v, built_v, 'cell15 v')
                _assert_bitwise(self, t, built_t, 'cell15 t')
                _assert_bitwise(self, current_input, built_input, 'cell15 current_input')
                _assert_bitwise(self, state_vectors, built_states, 'cell15 state_vectors')


if __name__ == '__main__':
    unittest.main()
