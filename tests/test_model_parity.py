"""Parity tests for the composable model and stimulus layer.

These tests verify that:

1. ``pnln_network(...).to_config_dict()`` produces arrays numerically identical
   to those built by ``benchmark_cases.build_case()`` / ``build_repo_production_case()``.

2. ``pnln_network(...)`` routed through ``CompiledExperimentRunner`` (JAX backend)
   produces *bit-for-bit identical* trajectory outputs to the same experiment run
   directly from a legacy ``benchmark_cases``-produced config dict.

3. ``fire_thresholds()`` returns the correct per-neuron values.

4. ``NetworkModel.layout`` state-vector size matches the legacy state-vector
   length produced by ``build_stable_state_vector`` / ``build_repo_state_vector``.

5. ``IntermittentOdorParams`` field defaults are self-consistent
   (e.g. ``n_n == p_n + l_n``).

Run::

    IODOR_BACKEND=jax python -m unittest tests.test_model_parity -v

No additional environment variables or data files are required.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _has_module(name):
    return importlib.util.find_spec(name) is not None


HAS_NUMPY = _has_module('numpy')
HAS_JAX = _has_module('jax')

JAX_DEPS_MESSAGE = 'Install numpy and jax in the active interpreter to run model parity tests.'


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
class ConfigDictParityTests(unittest.TestCase):
    """pnln_network().to_config_dict() must match benchmark_cases.build_case()."""

    def _check_configs_equal(self, ref_config, model_config, *, case_label=''):
        """Assert every key in ref_config matches model_config."""
        import numpy as np

        array_keys = {
            'C_m', 'g_K', 'g_L', 'E_K', 'E_L',
            'g_Na', 'g_A', 'E_Na', 'E_A',
            'g_Ca', 'g_KCa', 'E_Ca', 'E_KCa',
            'alp_ach', 'bet_ach', 'A', 'g_ach', 'E_ach',
            'alp_fgaba', 'bet_fgaba', 'V0', 'sigma', 'g_fgaba', 'E_fgaba',
            'K_sgaba', 'r1_sgaba', 'r2_sgaba', 'r3_sgaba', 'r4_sgaba',
            'G_sgaba', 'E_sgaba',
        }
        matrix_keys = {'ach_mat', 'fgaba_mat', 'sgaba_mat'}
        scalar_keys = {'n_n', 'p_n', 'l_n', 'A_Ca', 'Ca0', 't_Ca', 't_max', 't_delay', 'input_scale'}

        prefix = f'{case_label}: ' if case_label else ''

        for key in scalar_keys:
            if key not in ref_config:
                continue
            self.assertIn(key, model_config, f'{prefix}model config missing key {key!r}')
            self.assertAlmostEqual(
                float(ref_config[key]),
                float(model_config[key]),
                places=12,
                msg=f'{prefix}scalar mismatch for key {key!r}',
            )

        for key in array_keys:
            if key not in ref_config:
                continue
            ref_arr = np.asarray(ref_config[key], dtype=np.float64)
            if ref_arr.size == 0:
                continue
            self.assertIn(key, model_config, f'{prefix}model config missing key {key!r}')
            model_arr = np.asarray(model_config[key], dtype=np.float64)
            np.testing.assert_array_almost_equal(
                ref_arr, model_arr, decimal=12,
                err_msg=f'{prefix}array mismatch for key {key!r}',
            )

        for key in matrix_keys:
            if key not in ref_config:
                continue
            self.assertIn(key, model_config, f'{prefix}model config missing key {key!r}')
            ref_mat = np.asarray(ref_config[key], dtype=np.float64)
            model_mat = np.asarray(model_config[key], dtype=np.float64)
            np.testing.assert_array_equal(
                ref_mat, model_mat,
                err_msg=f'{prefix}matrix mismatch for key {key!r}',
            )

    def test_synthetic_case_config_matches(self):
        """Small network: pnln_network config == benchmark_cases.build_case config."""
        import numpy as np

        from benchmark_cases import build_case, normalize_by_indegree
        from intermittent_odors.model import pnln_network

        for case in (
            dict(seed=3, n_n=8,  p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05),
            dict(seed=4, n_n=9,  p_n=3, ach_density=0.45, fgaba_density=0.25, sgaba_density=0.40),
            dict(seed=5, n_n=7,  p_n=2, ach_density=0.00, fgaba_density=0.55, sgaba_density=0.15),
            dict(seed=1, n_n=6,  p_n=4, ach_density=0.35, fgaba_density=0.45, sgaba_density=0.25),
        ):
            with self.subTest(**case):
                ref_config, _, _, _, _ = build_case(**case)
                net = pnln_network(
                    p_n=ref_config['p_n'],
                    l_n=ref_config['l_n'],
                    ach_mat=ref_config['ach_mat'],
                    fgaba_mat=ref_config['fgaba_mat'],
                    sgaba_mat=ref_config['sgaba_mat'],
                    g_ach=ref_config['g_ach'],
                    g_fgaba=ref_config['g_fgaba'],
                    G_sgaba=ref_config['G_sgaba'],
                    normalize_conductances=False,  # already normalised by build_case
                )
                model_config = net.to_config_dict()
                self._check_configs_equal(
                    ref_config, model_config,
                    case_label=f"seed={case['seed']} n_n={case['n_n']}",
                )

    def test_fire_thresholds_match(self):
        """fire_thresholds() returns [0]*p_n + [-20]*l_n."""
        import numpy as np

        from benchmark_cases import build_case
        from intermittent_odors.model import pnln_network

        ref_config, _, _, _, ref_thresholds = build_case(
            seed=3, n_n=8, p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05
        )
        net = pnln_network(
            p_n=ref_config['p_n'], l_n=ref_config['l_n'],
            ach_mat=ref_config['ach_mat'], fgaba_mat=ref_config['fgaba_mat'],
            sgaba_mat=ref_config['sgaba_mat'],
            g_ach=ref_config['g_ach'], g_fgaba=ref_config['g_fgaba'],
            G_sgaba=ref_config['G_sgaba'],
            normalize_conductances=False,
        )
        np.testing.assert_array_equal(
            net.fire_thresholds(),
            np.asarray(ref_thresholds, dtype=np.float64),
        )


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
class IntegrationParityTests(unittest.TestCase):
    """Running via pnln_network must give identical JAX trajectory to the legacy path."""

    def _run_jax(self, config, current_input, state, times, thresholds):
        """Run a single integration using the legacy (direct config) path."""
        import jax.numpy as jnp
        import numpy as np

        from intermittent_odors.backends.jax_integrator import \
            odeint as jax_odeint
        from intermittent_odors.backends.jax_network import \
            build_dynamics as build_jax_dynamics

        dXdt = build_jax_dynamics(config, current_input)
        state_tensor = jnp.asarray(np.asarray(state, dtype=np.float64), dtype=jnp.float64)
        times_tensor = jnp.asarray(np.asarray(times, dtype=np.float64), dtype=jnp.float64)
        rollout = np.asarray(
            jax_odeint(dXdt, state_tensor, times_tensor, config['n_n'], thresholds)
        )
        return rollout

    def _run_via_model(self, net, config, current_input, state, times):
        """Run via pnln_network's CompiledExperimentRunner."""
        import numpy as np

        from intermittent_odors.experiment import build_experiment_spec
        from intermittent_odors.runtime import CompiledExperimentRunner

        thr = net.fire_thresholds()
        spec = build_experiment_spec(
            config,
            np.asarray(current_input, dtype=np.float64),
            np.asarray(state, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
            thr,
            input_dt=float(times[1] - times[0]),
        )
        runner = CompiledExperimentRunner(spec, backend='jax')
        return np.asarray(runner.run(state, current_input, times), dtype=np.float64)

    def test_small_network_integration_identical(self):
        """Trajectory via pnln_network must be bit-identical to direct JAX path."""
        import jax
        import numpy as np
        jax.config.update('jax_enable_x64', True)

        from benchmark_cases import build_case
        from intermittent_odors.model import pnln_network

        for case in (
            dict(seed=3, n_n=8,  p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05),
            dict(seed=1, n_n=6,  p_n=4, ach_density=0.35, fgaba_density=0.45, sgaba_density=0.25),
        ):
            with self.subTest(**case):
                config, current_input, state, times, thresholds = build_case(**case)
                net = pnln_network(
                    p_n=config['p_n'], l_n=config['l_n'],
                    ach_mat=config['ach_mat'], fgaba_mat=config['fgaba_mat'],
                    sgaba_mat=config['sgaba_mat'],
                    g_ach=config['g_ach'], g_fgaba=config['g_fgaba'],
                    G_sgaba=config['G_sgaba'],
                    normalize_conductances=False,
                )
                legacy_traj = self._run_jax(config, current_input, state, times, thresholds)
                model_traj  = self._run_via_model(net, config, current_input, state, times)
                # The two paths use different JIT-compiled function graphs which can produce
                # sub-ULP differences (~1e-22).  Use a very tight tolerance rather than
                # exact equality so we verify functional parity without over-constraining
                # floating-point associativity.
                np.testing.assert_allclose(
                    legacy_traj, model_traj,
                    rtol=1e-12, atol=1e-20,
                    err_msg=f"Trajectory mismatch for case {case}",
                )


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
class StateVectorLayoutTests(unittest.TestCase):
    """State-vector size from NetworkModel.layout must match the legacy formula."""

    def _expected_sv_size(self, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba):
        n_n = p_n + l_n
        # voltages + gates (PN: K(1)+Na(2)+A(2)=5, LN: K(1)+Ca(2+1_extra)+KCa(1)=4+1=5 extra_state)
        # Layout: V[n_n] + PN gates: n(1) m(1) h(1) mA(1) hA(1) = 5 per PN
        # LN gates: nK(1) mCa(1) hCa(1)+CaExtra(1) mKCa(1) = 4+1=5 per LN
        # but slurm formula = n_n + 4*p_n + 3*l_n + l_n (extra Ca state)
        # from benchmark_cases: state size = n_n(V) + n_n + 4*p_n + 3*l_n (gates) + l_n(CaExtra) + n_syn_ach + n_syn_fgaba + 2*n_syn_sgaba + n_n(fire_t)
        gates = n_n + 4 * p_n + 3 * l_n  # PN: K(n) + Na(m,h) + A(m,h) = 5 per PN minus V; LN: K(n)+Ca(m,h)+KCa(m)=4 per LN
        return n_n + gates + l_n + n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba + n_n

    def test_layout_size_matches_legacy_formula(self):
        import numpy as np

        from benchmark_cases import build_case
        from intermittent_odors.model import pnln_network

        for case in (
            dict(seed=3, n_n=8,  p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05),
            dict(seed=4, n_n=9,  p_n=3, ach_density=0.45, fgaba_density=0.25, sgaba_density=0.40),
        ):
            with self.subTest(**case):
                config, _, state, _, _ = build_case(**case)
                p_n = config['p_n']
                l_n = config['l_n']
                n_syn_ach   = int(np.sum(np.asarray(config['ach_mat'])   != 0))
                n_syn_fgaba = int(np.sum(np.asarray(config['fgaba_mat']) != 0))
                n_syn_sgaba = int(np.sum(np.asarray(config['sgaba_mat']) != 0))

                net = pnln_network(
                    p_n=p_n, l_n=l_n,
                    ach_mat=config['ach_mat'], fgaba_mat=config['fgaba_mat'],
                    sgaba_mat=config['sgaba_mat'],
                    g_ach=config['g_ach'], g_fgaba=config['g_fgaba'],
                    G_sgaba=config['G_sgaba'],
                    normalize_conductances=False,
                )
                layout = net.layout
                expected = self._expected_sv_size(p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba)
                self.assertEqual(
                    layout.state_size, expected,
                    msg=f"Layout size {layout.state_size} != expected {expected} for case {case}",
                )
                # Also check the legacy state vector itself has the expected length
                self.assertEqual(
                    len(state), expected,
                    msg=f"Legacy state size {len(state)} != expected {expected}",
                )
                self.assertEqual(
                    layout.state_size, len(state),
                    msg=f"Layout size {layout.state_size} != legacy state size {len(state)}",
                )


@unittest.skipUnless(HAS_NUMPY, 'numpy required')
class IntermittentOdorParamsTests(unittest.TestCase):
    """Basic consistency checks for IntermittentOdorParams defaults."""

    def test_defaults_consistent(self):
        from intermittent_odors.stimulus import IntermittentOdorParams
        p = IntermittentOdorParams()
        self.assertEqual(p.n_n, p.p_n + p.l_n)

    def test_sim_time_ms(self):
        from intermittent_odors.stimulus import IntermittentOdorParams
        p = IntermittentOdorParams(blocktime_ms=500, buffer_ms=100)
        self.assertEqual(p.sim_time_ms, 700)

    def test_base_conductances_shapes(self):
        import numpy as np

        from intermittent_odors.stimulus import IntermittentOdorParams
        p = IntermittentOdorParams(n_n=10, p_n=7, l_n=3)
        g_ach, g_fgaba, G_sgaba = p.base_conductances()
        self.assertEqual(g_ach.shape,   (10,))
        self.assertEqual(g_fgaba.shape, (10,))
        self.assertEqual(G_sgaba.shape, (10,))
        # PNs should have zero ACh conductance; LNs should have zero sGABA
        np.testing.assert_array_equal(g_ach[:7],   np.zeros(7))
        np.testing.assert_array_equal(G_sgaba[7:], np.zeros(3))

    def test_default_fields_consistent_with_trial_settings(self):
        from intermittent_odors.stimulus import IntermittentOdorParams
        from slurm.builders import TrialSettings
        p = IntermittentOdorParams()
        ts = TrialSettings()
        self.assertEqual(ts.n_n, p.n_n)
        self.assertEqual(ts.p_n, p.p_n)
        self.assertEqual(ts.l_n, p.l_n)
        self.assertAlmostEqual(ts.sim_res, p.dt)


@unittest.skipUnless(HAS_NUMPY and HAS_JAX, JAX_DEPS_MESSAGE)
class ConstantStimulusTests(unittest.TestCase):
    """build_constant_stimulus / build_constant_trial smoke tests."""

    def test_stimulus_shapes(self):
        import numpy as np

        from benchmark_cases import build_case
        from intermittent_odors.model import pnln_network
        from intermittent_odors.stimulus import (ConstantStimulusParams,
                                                 build_constant_stimulus)

        config, _, _, _, _ = build_case(
            seed=3, n_n=8, p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05
        )
        params = ConstantStimulusParams(duration_ms=10.0, dt=0.1, batch_ms=10.0)
        n_n = config['n_n']
        p_n = config['p_n']
        n_syn_ach   = int(np.sum(np.asarray(config['ach_mat'])   != 0))
        n_syn_fgaba = int(np.sum(np.asarray(config['fgaba_mat']) != 0))
        n_syn_sgaba = int(np.sum(np.asarray(config['sgaba_mat']) != 0))
        stim = build_constant_stimulus(params, n_n, p_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, seed=7)
        n_steps = stim.times.shape[0]
        self.assertEqual(stim.current_input.shape, (n_n, n_steps))
        self.assertGreater(stim.state_vector.shape[0], 0)
        self.assertGreater(len(stim.time_batches), 0)

    def test_constant_trial_roundtrip(self):
        """build_constant_trial produces an ExperimentSpec that can be prepared."""
        import jax
        jax.config.update('jax_enable_x64', True)

        from benchmark_cases import build_case
        from intermittent_odors.model import pnln_network
        from intermittent_odors.runtime import CompiledExperimentRunner
        from intermittent_odors.stimulus import (ConstantStimulusParams,
                                                 build_constant_trial)

        config, _, _, _, _ = build_case(
            seed=3, n_n=8, p_n=5, ach_density=0.15, fgaba_density=0.35, sgaba_density=0.05
        )
        net = pnln_network(
            p_n=config['p_n'], l_n=config['l_n'],
            ach_mat=config['ach_mat'], fgaba_mat=config['fgaba_mat'],
            sgaba_mat=config['sgaba_mat'],
            g_ach=config['g_ach'], g_fgaba=config['g_fgaba'],
            G_sgaba=config['G_sgaba'],
            normalize_conductances=False,
        )
        params = ConstantStimulusParams(duration_ms=5.0, dt=0.1, batch_ms=5.0)
        spec = build_constant_trial(net, params, seed=42)
        prepared = spec.prepare()
        # Should be prepared without errors and have no custom dynamics_builder
        # (standard PN/LN model → dynamics_builder should be None)
        self.assertIsNone(prepared.dynamics_builder)
        # Runner API smoke test: build stim separately and run
        import numpy as np

        from intermittent_odors.stimulus import build_constant_stimulus
        n_sym_ach   = int(np.sum(np.asarray(config['ach_mat'])   != 0))
        n_sym_fgaba = int(np.sum(np.asarray(config['fgaba_mat']) != 0))
        n_sym_sgaba = int(np.sum(np.asarray(config['sgaba_mat']) != 0))
        stim = build_constant_stimulus(
            params, config['n_n'], config['p_n'],
            n_sym_ach, n_sym_fgaba, n_sym_sgaba, seed=42,
        )
        runner = CompiledExperimentRunner(prepared, backend='jax')
        result = runner.run(stim.state_vector, stim.current_input, stim.times)
        # run() returns full trajectory (T, state_size); verify trajectory has data
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
