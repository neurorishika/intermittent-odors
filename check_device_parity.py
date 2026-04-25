import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CPU_PYTHON = ROOT / '.venv' / 'bin' / 'python'
DEFAULT_GPU_PYTHON = ROOT / '.venv-gpu-bench' / 'bin' / 'python'


def max_abs_diff(left, right):
    matching_special = (
        (np.isnan(left) & np.isnan(right))
        | (np.isposinf(left) & np.isposinf(right))
        | (np.isneginf(left) & np.isneginf(right))
    )
    valid = ~matching_special
    if not np.any(valid):
        return 0.0
    diff = np.abs(np.nan_to_num(left[valid] - right[valid], nan=np.inf, posinf=np.inf, neginf=np.inf))
    return float(np.max(diff))


def build_experiment_case(case_name):
    from benchmark_cases import (build_case, build_realistic_trial_case,
                                 build_repo_production_case)
    from intermittent_odors.experiment import (build_experiment_spec,
                                               infer_input_dt_from_times)

    if case_name == 'synthetic':
        config, current_input, state_vector, times, thresholds = build_case(
            seed=3,
            n_n=8,
            p_n=5,
            ach_density=0.15,
            fgaba_density=0.35,
            sgaba_density=0.05,
        )
        experiment = build_experiment_spec(
            config,
            current_input,
            state_vector,
            times,
            thresholds,
            input_dt=infer_input_dt_from_times(times),
            sample_stride=1,
            sample_neurons=config['n_n'],
            time_batches=(times,),
            metadata={'case': case_name, 'family': 'device-parity'},
            network_metadata={'family': 'device-parity'},
            stimulus_metadata={'family': 'device-parity'},
        )
        return {
            'mode': 'trajectory',
            'experiment': experiment,
            'current_input': current_input,
            'state_vector': state_vector,
            'times': times,
        }

    if case_name == 'repo-production':
        config, current_input, state_vector, times, thresholds = build_repo_production_case(
            graph_no=1,
            odor_seed=59428,
            trial_seed=1,
        )
        experiment = build_experiment_spec(
            config,
            current_input,
            state_vector,
            times,
            thresholds,
            input_dt=infer_input_dt_from_times(times),
            sample_stride=1,
            sample_neurons=config['n_n'],
            time_batches=(times,),
            metadata={'case': case_name, 'family': 'device-parity'},
            network_metadata={'family': 'device-parity'},
            stimulus_metadata={'family': 'device-parity'},
        )
        return {
            'mode': 'trajectory',
            'experiment': experiment,
            'current_input': current_input,
            'state_vector': state_vector,
            'times': times,
        }

    if case_name == 'realistic-slurm':
        case_plan = build_realistic_trial_case(
            total_neurons=120,
            p_neurons=90,
            graph_no=1,
            odor_seed=59428,
            trial_seed=1,
            blocktime_ms=200,
            buffer_ms=50,
            sim_res_ms=0.01,
            batch_ms=100,
            switch_prob=0.1,
            sample_every_ms=1.0,
        )
        return {
            'mode': 'time-batches',
            'experiment': case_plan['experiment_spec'],
            'current_input': case_plan['current_input'],
            'state_vector': case_plan['state_vector'],
            'time_batches': tuple(case_plan['time_batches']),
        }

    raise ValueError(f'Unsupported case {case_name!r}.')


def run_worker(case_name, output_path):
    from intermittent_odors.runtime import compile_experiment

    case = build_experiment_case(case_name)
    runner = compile_experiment(case['experiment'], backend='jax')

    if case['mode'] == 'trajectory':
        output = runner.run(case['state_vector'], case['current_input'], case['times'])
        final_state = np.asarray(output[-1], dtype=np.float64)
    else:
        output, final_state = runner.run_time_batches(
            case['state_vector'],
            case['current_input'],
            time_batches=case['time_batches'],
        )

    np.savez(
        output_path,
        output=np.asarray(output, dtype=np.float64),
        final_state=np.asarray(final_state, dtype=np.float64),
    )


def resolve_python(path, fallback):
    if path:
        return Path(path)
    return fallback if fallback.exists() else Path(sys.executable)


def query_jax_runtime(python_path, platform):
    env = os.environ.copy()
    env['JAX_PLATFORM_NAME'] = platform
    env['IODOR_JAX_PRECISION'] = 'float64'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    command = [
        str(python_path),
        '-c',
        (
            'import json, jax; '
            'print(json.dumps({'
            '"default_backend": jax.default_backend(), '
            '"devices": [str(device) for device in jax.devices()]'
            '}))'
        ),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Failed to inspect JAX runtime for {python_path} on {platform}.\n'
            f'stdout:\n{completed.stdout}\n'
            f'stderr:\n{completed.stderr}'
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def run_case(python_path, platform, case_name, output_path):
    env = os.environ.copy()
    env['IODOR_BACKEND'] = 'jax'
    env['IODOR_JAX_PRECISION'] = 'float64'
    env['JAX_PLATFORM_NAME'] = platform
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env.setdefault('MPLBACKEND', 'Agg')
    command = [
        str(python_path),
        str(Path(__file__).resolve()),
        '--worker',
        '--case',
        case_name,
        '--output',
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f'{case_name} failed for platform={platform} using {python_path}.\n'
            f'stdout:\n{completed.stdout}\n'
            f'stderr:\n{completed.stderr}'
        )
    with np.load(output_path) as payload:
        return {
            'output': np.asarray(payload['output'], dtype=np.float64),
            'final_state': np.asarray(payload['final_state'], dtype=np.float64),
        }


def compare_case(case_name, cpu_payload, gpu_payload, atol, rtol):
    output_diff = max_abs_diff(cpu_payload['output'], gpu_payload['output'])
    final_state_diff = max_abs_diff(cpu_payload['final_state'], gpu_payload['final_state'])
    print(f'{case_name} output: max abs diff = {output_diff:.3e}')
    print(f'{case_name} final state: max abs diff = {final_state_diff:.3e}')
    if not np.allclose(cpu_payload['output'], gpu_payload['output'], atol=atol, rtol=rtol, equal_nan=True):
        raise SystemExit(f'{case_name} output parity check failed')
    if not np.allclose(cpu_payload['final_state'], gpu_payload['final_state'], atol=atol, rtol=rtol, equal_nan=True):
        raise SystemExit(f'{case_name} final state parity check failed')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Compare JAX CPU and GPU outputs for representative intermittent-odors cases.')
    parser.add_argument('--worker', action='store_true', help='Internal worker mode.')
    parser.add_argument('--case', choices=['synthetic', 'repo-production', 'realistic-slurm'])
    parser.add_argument('--output')
    parser.add_argument('--cpu-python')
    parser.add_argument('--gpu-python')
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--rtol', type=float, default=1e-10)
    parser.add_argument('--skip-realistic-slurm', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if not args.case or not args.output:
            raise SystemExit('--case and --output are required in worker mode.')
        run_worker(args.case, args.output)
        return 0

    cpu_python = resolve_python(args.cpu_python, DEFAULT_CPU_PYTHON)
    gpu_python = resolve_python(args.gpu_python, DEFAULT_GPU_PYTHON)
    if not cpu_python.exists():
        raise SystemExit(f'CPU Python interpreter not found: {cpu_python}')
    if not gpu_python.exists():
        raise SystemExit(f'GPU Python interpreter not found: {gpu_python}')

    cpu_runtime = query_jax_runtime(cpu_python, 'cpu')
    gpu_runtime = query_jax_runtime(gpu_python, 'gpu')
    print(f'cpu_python={cpu_python}')
    print(f'cpu_devices={cpu_runtime["devices"]}')
    print(f'gpu_python={gpu_python}')
    print(f'gpu_devices={gpu_runtime["devices"]}')

    if cpu_runtime['default_backend'] != 'cpu':
        raise SystemExit(f'Expected CPU backend for {cpu_python}, got {cpu_runtime["default_backend"]!r}.')
    if gpu_runtime['default_backend'] != 'gpu':
        raise SystemExit(f'Expected GPU backend for {gpu_python}, got {gpu_runtime["default_backend"]!r}.')

    cases = ['synthetic', 'repo-production']
    if not args.skip_realistic_slurm:
        cases.append('realistic-slurm')

    with tempfile.TemporaryDirectory(prefix='iodor-device-parity-') as temp_dir:
        temp_root = Path(temp_dir)
        for case_name in cases:
            cpu_output = temp_root / f'{case_name}_cpu.npz'
            gpu_output = temp_root / f'{case_name}_gpu.npz'
            cpu_payload = run_case(cpu_python, 'cpu', case_name, cpu_output)
            gpu_payload = run_case(gpu_python, 'gpu', case_name, gpu_output)
            compare_case(case_name, cpu_payload, gpu_payload, args.atol, args.rtol)

    print('JAX CPU/GPU parity checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())