import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _has_module(name):
    return importlib.util.find_spec(name) is not None


HAS_NUMPY = _has_module('numpy')
HAS_TENSORFLOW = _has_module('tensorflow')
HAS_JAX = _has_module('jax')
RUN_SCRIPT_PARITY = os.environ.get('IODOR_RUN_SCRIPT_PARITY') == '1'
RUN_DEVICE_PARITY = os.environ.get('IODOR_RUN_DEVICE_PARITY') == '1'
RUN_NOTEBOOK_NUMERICAL_EQUIVALENCE = os.environ.get('IODOR_RUN_NOTEBOOK_NUMERICAL_EQUIVALENCE') == '1'
HAS_NOTEBOOK_NUMERICAL_INPUTS = all(
    path.exists()
    for path in (
        ROOT / 'data' / 'zip_0_0.zip',
        ROOT / 'data' / 'zip_0_3.zip',
        ROOT / 'data' / '30LN' / 'LN30_events_1_59428.npy',
    )
)

TF_DEPS_MESSAGE = 'Install numpy and tensorflow in the active interpreter to run TensorFlow parity tests.'
JAX_DEPS_MESSAGE = 'Install numpy, tensorflow, and jax in the active interpreter to run JAX parity tests.'
SCRIPT_PARITY_MESSAGE = 'Set IODOR_RUN_SCRIPT_PARITY=1 to include the slower fig2/slurm script-level parity checks.'
DEVICE_PARITY_MESSAGE = 'Set IODOR_RUN_DEVICE_PARITY=1 to include the slower JAX CPU/GPU parity checks.'
NOTEBOOK_NUMERICAL_MESSAGE = 'Set IODOR_RUN_NOTEBOOK_NUMERICAL_EQUIVALENCE=1 to include slower notebook numerical artifact checks.'
NOTEBOOK_NUMERICAL_DATA_MESSAGE = 'Notebook numerical artifact checks require the extracted external datasets under data/.'
GPU_PYTHON = ROOT / '.venv-gpu-bench' / 'bin' / 'python'
DEVICE_PARITY_ENV_MESSAGE = 'Device parity checks require .venv-gpu-bench/bin/python with a visible JAX GPU backend.'

EXTRA_SYNTHETIC_CASES = (
    {'seed': 3, 'n_n': 8, 'p_n': 5, 'ach_density': 0.15, 'fgaba_density': 0.35, 'sgaba_density': 0.05},
    {'seed': 4, 'n_n': 9, 'p_n': 3, 'ach_density': 0.45, 'fgaba_density': 0.25, 'sgaba_density': 0.4},
    {'seed': 5, 'n_n': 7, 'p_n': 2, 'ach_density': 0.0, 'fgaba_density': 0.55, 'sgaba_density': 0.15},
)

EXTRA_PRODUCTION_CASES = (
    {'graph_no': 1, 'odor_seed': 13674, 'trial_seed': 2},
    {'graph_no': 2, 'odor_seed': 59428, 'trial_seed': 3},
)


@unittest.skipUnless(HAS_NUMPY and HAS_TENSORFLOW, TF_DEPS_MESSAGE)
class SharedTensorFlowParityTests(unittest.TestCase):
    def test_shared_core_matches_legacy_tensorflow(self):
        import check_shared_tf_parity

        check_shared_tf_parity.main()

    def test_shared_core_matches_legacy_tensorflow_additional_cases(self):
        import check_shared_tf_parity

        for case in EXTRA_SYNTHETIC_CASES:
            with self.subTest(kind='synthetic', **case):
                check_shared_tf_parity.run_case(
                    f"synthetic_{case['seed']}",
                    *check_shared_tf_parity.build_case(**case),
                )

        for case in EXTRA_PRODUCTION_CASES:
            with self.subTest(kind='production', **case):
                check_shared_tf_parity.run_case(
                    f"production_{case['graph_no']}_{case['odor_seed']}_{case['trial_seed']}",
                    *check_shared_tf_parity.build_repo_production_case(**case),
                )


@unittest.skipUnless(HAS_NUMPY and HAS_TENSORFLOW and HAS_JAX, JAX_DEPS_MESSAGE)
class JaxParityTests(unittest.TestCase):
    def test_shared_jax_core_matches_tensorflow(self):
        import check_jax_parity

        check_jax_parity.main()

    def test_shared_jax_core_matches_tensorflow_additional_cases(self):
        import check_jax_parity
        import check_shared_tf_parity

        for case in EXTRA_SYNTHETIC_CASES:
            with self.subTest(kind='synthetic', **case):
                check_jax_parity.run_case(
                    f"synthetic_{case['seed']}",
                    *check_shared_tf_parity.build_case(**case),
                )

        for case in EXTRA_PRODUCTION_CASES:
            with self.subTest(kind='production', **case):
                check_jax_parity.run_case(
                    f"production_{case['graph_no']}_{case['odor_seed']}_{case['trial_seed']}",
                    *check_shared_tf_parity.build_repo_production_case(**case),
                )

    @unittest.skipUnless(RUN_SCRIPT_PARITY, SCRIPT_PARITY_MESSAGE)
    def test_script_entrypoints_match_between_backends(self):
        import check_script_jax_parity

        check_script_jax_parity.main()

    @unittest.skipUnless(RUN_DEVICE_PARITY, DEVICE_PARITY_MESSAGE)
    @unittest.skipUnless(GPU_PYTHON.exists(), DEVICE_PARITY_ENV_MESSAGE)
    def test_jax_cpu_gpu_device_parity(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / 'check_device_parity.py'),
                '--gpu-python',
                str(GPU_PYTHON),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.fail(
                'check_device_parity.py failed\n'
                f'stdout:\n{completed.stdout}\n'
                f'stderr:\n{completed.stderr}'
            )

    @unittest.skipUnless(RUN_NOTEBOOK_NUMERICAL_EQUIVALENCE, NOTEBOOK_NUMERICAL_MESSAGE)
    @unittest.skipUnless(HAS_NOTEBOOK_NUMERICAL_INPUTS, NOTEBOOK_NUMERICAL_DATA_MESSAGE)
    def test_notebook_numerical_artifacts_match(self):
        import check_notebook_regeneration

        exit_code = check_notebook_regeneration.main([
            '--numerical-only',
            '--notebook', 'fig4/fig4.ipynb',
            '--notebook', 'fig5_6/fig5_6.ipynb',
            '--notebook', 'fig7/fig7.ipynb',
            '--notebook', 'fig8/fig8.ipynb',
        ])
        self.assertEqual(exit_code, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)