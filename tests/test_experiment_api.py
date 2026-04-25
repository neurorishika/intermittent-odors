import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intermittent_odors.builders import (TrialCase, TrialSettings,
                                         build_fig2_experiment_spec,
                                         piecewise_profile,
                                         trial_case_to_experiment_spec)
from intermittent_odors.experiment import (build_experiment_spec,
                                           ensure_prepared_experiment,
                                           prepare_experiment)


def _build_config(fgaba_scale=1.0):
    ach_mat = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    fgaba_mat = np.array([[0.0, 0.0], [fgaba_scale, 0.0]], dtype=np.float64)
    sgaba_mat = np.zeros((2, 2), dtype=np.float64)
    return {
        'n_n': 2,
        'p_n': 1,
        'l_n': 1,
        'C_m': [1.0, 1.0],
        'g_K': [3.6, 36.0],
        'g_L': [0.3, 0.3],
        'E_K': [-95.0, -95.0],
        'E_L': [-64.0, -50.0],
        'g_Na': [7.15],
        'g_A': [1.43],
        'E_Na': [50.0],
        'E_A': [-95.0],
        'g_Ca': [5.0],
        'g_KCa': [0.045],
        'E_Ca': [140.0],
        'E_KCa': [-95.0],
        'A_Ca': 2e-4,
        'Ca0': 2.4e-4,
        't_Ca': 150.0,
        'ach_mat': ach_mat,
        'alp_ach': [10.0],
        'bet_ach': [0.2],
        't_max': 0.3,
        't_delay': 0.0,
        'A': [0.5, 0.5],
        'g_ach': [0.0, 0.225],
        'E_ach': [0.0, 0.0],
        'fgaba_mat': fgaba_mat,
        'alp_fgaba': [10.0],
        'bet_fgaba': [0.16],
        'V0': [-20.0, -20.0],
        'sigma': [1.5, 1.5],
        'g_fgaba': [2.16, 3.6],
        'E_fgaba': [-70.0, -70.0],
        'sgaba_mat': sgaba_mat,
        'K_sgaba': [],
        'r1_sgaba': [],
        'r2_sgaba': [],
        'r3_sgaba': [],
        'r4_sgaba': [],
        'G_sgaba': [0.0, 0.0],
        'E_sgaba': [-95.0, -95.0],
    }


class PreparedExperimentApiTests(unittest.TestCase):
    def test_static_model_digest_changes_when_values_change(self):
        thresholds = [0.0, -20.0]
        left = prepare_experiment(_build_config(fgaba_scale=1.0), thresholds, input_dt=0.02)
        right = prepare_experiment(_build_config(fgaba_scale=0.0), thresholds, input_dt=0.02)

        self.assertNotEqual(left.model_digest, right.model_digest)

    def test_prepare_experiment_precomputes_synapse_layouts(self):
        experiment = prepare_experiment(_build_config(), [0.0, -20.0], input_dt=0.02)

        self.assertEqual(experiment.config['n_syn_ach'], 1)
        self.assertEqual(experiment.config['n_syn_fgaba'], 1)
        self.assertEqual(experiment.config['n_syn_sgaba'], 0)
        self.assertAlmostEqual(experiment.config['input_scale'], 50.0)
        np.testing.assert_array_equal(experiment.config['ach_row_ids'], np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(experiment.config['ach_col_ids'], np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(experiment.config['fgaba_row_ids'], np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(experiment.config['fgaba_col_ids'], np.array([0], dtype=np.int32))

    def test_ensure_prepared_experiment_overrides_sampling(self):
        base = prepare_experiment(_build_config(), [0.0, -20.0], input_dt=0.01, sample_stride=1, sample_neurons=2)
        updated = ensure_prepared_experiment(base, sample_stride=4, sample_neurons=1)

        self.assertEqual(updated.sample_stride, 4)
        self.assertEqual(updated.sample_neurons, 1)
        self.assertEqual(base.sample_stride, 1)
        self.assertEqual(base.sample_neurons, 2)

    def test_build_experiment_spec_round_trips_to_prepared_experiment(self):
        config = _build_config()
        current_input = np.zeros((2, 4), dtype=np.float64)
        state_vector = np.zeros(12, dtype=np.float64)
        times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float64)

        experiment_spec = build_experiment_spec(
            config,
            current_input,
            state_vector,
            times,
            [0.0, -20.0],
            sample_stride=2,
            sample_neurons=2,
            time_batches=(times,),
            metadata={'family': 'unit-test'},
        )
        prepared = ensure_prepared_experiment(experiment_spec)

        self.assertEqual(experiment_spec.network.n_n, 2)
        self.assertEqual(experiment_spec.sample_stride, 2)
        self.assertEqual(prepared.sample_stride, 2)
        self.assertEqual(prepared.sample_neurons, 2)
        np.testing.assert_array_equal(prepared.thresholds, np.array([0.0, -20.0], dtype=np.float64))

    def test_trial_case_to_experiment_spec_builds_slurm_experiment(self):
        case = TrialCase(
            ach_mat=np.zeros((2, 2), dtype=np.float64),
            fgaba_mat=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            sgaba_mat=np.zeros((2, 2), dtype=np.float64),
            current_input=np.zeros((2, 4), dtype=np.float64),
            state_vector=np.zeros(12, dtype=np.float64),
            times=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64),
            time_batches=(np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64),),
        )
        settings = TrialSettings(n_n=2, p_n=1, l_n=1, sim_res=0.5)

        experiment_spec = trial_case_to_experiment_spec(case, settings, metadata={'family': 'unit-test'})
        prepared = ensure_prepared_experiment(experiment_spec)

        self.assertEqual(experiment_spec.network.n_n, 2)
        self.assertEqual(experiment_spec.sample_stride, 2)
        self.assertEqual(prepared.sample_stride, 2)
        np.testing.assert_array_equal(prepared.config['fgaba_mat'], case.fgaba_mat)

    def test_fig2_builder_returns_experiment_spec(self):
        metadata = {
            'n_n': 2,
            'p_n': 1,
            'l_n': 1,
            'sim_res': 0.5,
            'fgaba_mat': np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        }
        experiment_spec = build_fig2_experiment_spec(
            metadata,
            np.zeros((2, 4), dtype=np.float64),
            np.zeros(12, dtype=np.float64),
            np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64),
            g_fgaba=piecewise_profile(1, 1, 0.0, 1.0),
        )

        prepared = ensure_prepared_experiment(experiment_spec)
        self.assertEqual(experiment_spec.sample_stride, 2)
        self.assertEqual(prepared.sample_stride, 2)
        np.testing.assert_array_equal(prepared.config['fgaba_mat'], metadata['fgaba_mat'])


if __name__ == '__main__':
    unittest.main(verbosity=2)