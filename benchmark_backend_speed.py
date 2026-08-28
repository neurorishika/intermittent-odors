import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

MODE_SPECS = (
    {'label': 'tensorflow', 'backend': 'tensorflow', 'precision': None},
    {'label': 'jax-float64', 'backend': 'jax', 'precision': 'float64'},
    {'label': 'jax-float32', 'backend': 'jax', 'precision': 'float32'},
    {'label': 'jax-bfloat16', 'backend': 'jax', 'precision': 'bfloat16'},
)


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


def build_case_plan(args):
    from benchmark_cases import build_case as build_synthetic_case
    from benchmark_cases import (build_realistic_trial_case,
                                 build_repo_production_case)
    from intermittent_odors.experiment import build_experiment_spec

    if args.case == 'realistic-slurm':
        return build_realistic_trial_case(
            total_neurons=args.n_neurons,
            p_neurons=args.p_neurons,
            graph_no=args.graph_no,
            odor_seed=args.odor_seed,
            trial_seed=args.trial_seed,
            blocktime_ms=args.blocktime_ms,
            buffer_ms=args.buffer_ms,
            sim_res_ms=args.sim_res_ms,
            batch_ms=args.chunk_ms,
            switch_prob=args.switch_prob,
            sample_every_ms=args.sample_every_ms,
        )

    if args.case == 'repo-production':
        config, current_input, state, times, thresholds = build_repo_production_case(
            graph_no=args.graph_no,
            odor_seed=args.odor_seed,
            trial_seed=args.trial_seed,
        )
    elif args.case == 'synthetic':
        config, current_input, state, times, thresholds = build_synthetic_case(
            seed=3,
            n_n=8,
            p_n=5,
            ach_density=0.15,
            fgaba_density=0.35,
            sgaba_density=0.05,
        )
    else:
        raise ValueError(f'Unsupported case {args.case!r}.')

    dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    experiment_spec = build_experiment_spec(
        config,
        current_input,
        state,
        times,
        thresholds,
        input_dt=dt if dt else None,
        sample_stride=1,
        sample_neurons=None,
        time_batches=[times],
        metadata={'topology': 'direct', 'case': args.case},
        network_metadata={'family': args.case},
        stimulus_metadata={'family': args.case},
    )
    return {
        'experiment_spec': experiment_spec,
        'config': config,
        'current_input': current_input,
        'state_vector': state,
        'times': times,
        'time_batches': [times],
        'thresholds': thresholds,
        'sample_stride': 1,
        'sample_neurons': None,
        'simulated_ms': float(times[-1] + dt) if len(times) else 0.0,
        'topology': 'direct',
    }


def build_case_batch(case_plan, batch_size):
    config = case_plan['config']
    current_input = case_plan['current_input']
    state = case_plan['state_vector']
    if batch_size == 1:
        return config, current_input[None, ...], state[None, ...]

    rng = np.random.default_rng(20260425)
    current_inputs = np.repeat(current_input[None, ...], batch_size, axis=0)
    state_vectors = np.repeat(state[None, ...], batch_size, axis=0)

    current_inputs += 0.01 * current_inputs * rng.normal(size=current_inputs.shape)
    current_inputs += 1e-4 * rng.normal(size=current_inputs.shape)
    state_vectors[:, : config['n_n']] += 0.25 * rng.normal(size=(batch_size, config['n_n']))
    state_vectors[:, config['n_n']:] += 0.01 * rng.normal(size=state_vectors[:, config['n_n']:].shape)
    return config, current_inputs, state_vectors


def run_chunked_rollout(case_plan, runner, current_inputs, state_vectors, batch_size):
    if batch_size == 1:
        output, _ = runner.run_time_batches(state_vectors[0], current_inputs[0])
        return output

    output, _ = runner.run_time_batches_batch(state_vectors, current_inputs)
    return output


def run_worker(args):
    os.environ.pop('IODOR_JAX_PRECISION', None)
    if args.precision:
        os.environ['IODOR_JAX_PRECISION'] = args.precision
    if args.platform:
        os.environ['JAX_PLATFORM_NAME'] = args.platform
    if args.jax_preallocate is not None:
        os.environ['IODOR_JAX_PREALLOCATE'] = args.jax_preallocate
    if args.jax_mem_fraction is not None:
        os.environ['IODOR_JAX_MEM_FRACTION'] = str(args.jax_mem_fraction)
    if args.jax_allocator:
        os.environ['IODOR_JAX_ALLOCATOR'] = args.jax_allocator
    if args.xla_flags_append:
        os.environ['IODOR_XLA_FLAGS_APPEND'] = ' '.join(args.xla_flags_append)

    from slurm.builders import configure_runtime_environment

    os.environ.update(configure_runtime_environment(ROOT, os.environ.copy()))

    from intermittent_odors.experiment import (build_experiment_spec,
                                               infer_input_dt_from_times)
    from intermittent_odors.runtime import compile_experiment

    case_plan = build_case_plan(args)
    config, current_inputs, state_vectors = build_case_batch(case_plan, args.batch_size)
    experiment = case_plan.get('experiment_spec')
    if experiment is None:
        experiment = build_experiment_spec(
            config,
            case_plan['current_input'],
            case_plan['state_vector'],
            case_plan['times'],
            case_plan['thresholds'],
            input_dt=infer_input_dt_from_times(case_plan['times']),
            sample_stride=case_plan['sample_stride'],
            sample_neurons=case_plan['sample_neurons'],
            time_batches=case_plan['time_batches'],
            metadata={'topology': case_plan['topology'], 'case': args.case},
            network_metadata={'family': case_plan['topology']},
            stimulus_metadata={'family': case_plan['topology']},
        )
    runner = compile_experiment(experiment, backend=args.backend)

    timings = []
    output = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        if args.case == 'realistic-slurm':
            output = run_chunked_rollout(
                case_plan,
                runner,
                current_inputs,
                state_vectors,
                args.batch_size,
            )
        elif args.batch_size == 1:
            output = runner.run(state_vectors[0], current_inputs[0], case_plan['times'])
        else:
            output = runner.run_batch(state_vectors, current_inputs, case_plan['times'])
        timings.append(time.perf_counter() - start)

    np.save(args.output, output)
    total_simulated_ms = float(case_plan['simulated_ms']) * args.batch_size
    neuron_steps = int(config['n_n'] * len(case_plan['times']) * args.batch_size)
    warm_seconds = min(timings[1:]) if len(timings) > 1 else timings[0]
    payload = {
        'label': args.label,
        'backend': args.backend,
        'precision': args.precision,
        'platform': args.platform,
        'jax_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        'jax_mem_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
        'jax_allocator': os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR'),
        'xla_flags': os.environ.get('XLA_FLAGS'),
        'repeats': args.repeats,
        'batch_size': args.batch_size,
        'timings_seconds': timings,
        'cold_seconds': timings[0],
        'warm_seconds': warm_seconds,
        'output_shape': list(output.shape),
        'output_max_abs': float(np.max(np.abs(output))),
        'all_finite': bool(np.isfinite(output).all()),
        'n_neurons': int(config['n_n']),
        'simulated_ms': total_simulated_ms,
        'simulated_ms_per_second': total_simulated_ms / warm_seconds if warm_seconds else float('inf'),
        'million_neuron_steps_per_second': (neuron_steps / warm_seconds) / 1e6 if warm_seconds else float('inf'),
        'topology': case_plan['topology'],
    }
    print(json.dumps(payload))


def run_mode(mode, args, output_dir):
    output_path = output_dir / f"{mode['label']}_{args.platform}_{args.case}_n{args.n_neurons}_batch{args.batch_size}.npy"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--worker',
        '--label', mode['label'],
        '--backend', mode['backend'],
        '--case', args.case,
        '--repeats', str(args.repeats),
        '--platform', args.platform,
        '--batch-size', str(args.batch_size),
        '--output', str(output_path),
        '--graph-no', str(args.graph_no),
        '--odor-seed', str(args.odor_seed),
        '--trial-seed', str(args.trial_seed),
        '--n-neurons', str(args.n_neurons),
        '--blocktime-ms', str(args.blocktime_ms),
        '--buffer-ms', str(args.buffer_ms),
        '--sim-res-ms', str(args.sim_res_ms),
        '--chunk-ms', str(args.chunk_ms),
        '--switch-prob', str(args.switch_prob),
        '--sample-every-ms', str(args.sample_every_ms),
    ]
    if args.p_neurons is not None:
        command.extend(['--p-neurons', str(args.p_neurons)])
    if mode['precision']:
        command.extend(['--precision', mode['precision']])
    if args.jax_preallocate is not None:
        command.extend(['--jax-preallocate', args.jax_preallocate])
    if args.jax_mem_fraction is not None:
        command.extend(['--jax-mem-fraction', str(args.jax_mem_fraction)])
    if args.jax_allocator:
        command.extend(['--jax-allocator', args.jax_allocator])
    for value in args.xla_flags_append:
        command.extend(['--xla-flags-append', value])

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            'label': mode['label'],
            'backend': mode['backend'],
            'precision': mode['precision'],
            'platform': args.platform,
            'error': completed.stderr.strip() or completed.stdout.strip() or 'worker failed',
        }

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    payload['output_path'] = str(output_path)
    return payload


def print_summary(results, baseline):
    print('label\tplatform\tcold_s\twarm_s\tsim_ms_per_s\tMneuron_steps_per_s\tmax_abs_diff\tfinite\tstatus')
    for result in results:
        if 'error' in result:
            print(f"{result['label']}\t{result['platform']}\t-\t-\t-\t-\t-\t-\tERROR: {result['error']}")
            continue

        benchmark_output = np.load(result['output_path'])
        baseline_output = np.load(baseline['output_path'])
        diff = max_abs_diff(baseline_output, benchmark_output)
        print(
            f"{result['label']}\t{result['platform']}\t"
            f"{result['cold_seconds']:.6f}\t{result['warm_seconds']:.6f}\t"
            f"{result['simulated_ms_per_second']:.1f}\t{result['million_neuron_steps_per_second']:.3f}\t"
            f"{diff:.3e}\t"
            f"{result['all_finite']}\tOK"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Benchmark backend speed and parity for the intermittent-odors simulation path.')
    parser.add_argument('--worker', action='store_true', help='Internal worker mode.')
    parser.add_argument('--label', default='worker')
    parser.add_argument('--backend', choices=['tensorflow', 'jax'], default='jax')
    parser.add_argument('--precision', choices=['float64', 'float32', 'bfloat16'])
    parser.add_argument('--platform', default='cpu', help='JAX platform name, typically cpu or gpu.')
    parser.add_argument('--case', choices=['repo-production', 'synthetic', 'realistic-slurm'], default='repo-production')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--graph-no', type=int, default=1)
    parser.add_argument('--odor-seed', type=int, default=59428)
    parser.add_argument('--trial-seed', type=int, default=1)
    parser.add_argument('--n-neurons', type=int, default=120)
    parser.add_argument('--p-neurons', type=int)
    parser.add_argument('--blocktime-ms', type=int, default=1000)
    parser.add_argument('--buffer-ms', type=int, default=100)
    parser.add_argument('--sim-res-ms', type=float, default=0.01)
    parser.add_argument('--chunk-ms', type=int, default=250)
    parser.add_argument('--switch-prob', type=float, default=0.1)
    parser.add_argument('--sample-every-ms', type=float, default=1.0)
    parser.add_argument('--jax-preallocate', choices=['true', 'false'])
    parser.add_argument('--jax-mem-fraction', type=float)
    parser.add_argument('--jax-allocator')
    parser.add_argument('--xla-flags-append', action='append', default=[])
    parser.add_argument('--output')
    parser.add_argument('--skip-tensorflow', action='store_true', help='Skip the TensorFlow reference run.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if not args.output:
            raise SystemExit('--output is required in worker mode.')
        run_worker(args)
        return 0

    modes = [mode for mode in MODE_SPECS if not (args.skip_tensorflow and mode['backend'] == 'tensorflow')]
    with tempfile.TemporaryDirectory(prefix='iodor-benchmark-') as temp_dir:
        output_dir = Path(temp_dir)
        results = [run_mode(mode, args, output_dir) for mode in modes]

        baseline = next((result for result in results if result.get('label') == 'tensorflow' and 'error' not in result), None)
        if baseline is None:
            baseline = next((result for result in results if result.get('label') == 'jax-float64' and 'error' not in result), None)
        if baseline is None:
            print_summary(results, results[0])
            raise SystemExit('No successful baseline run was produced.')

        print(f'benchmark_case={args.case}')
        if args.case == 'realistic-slurm':
            print(f'n_neurons={args.n_neurons}')
            if args.p_neurons is not None:
                print(f'p_neurons={args.p_neurons}')
            print(f'blocktime_ms={args.blocktime_ms}')
            print(f'buffer_ms={args.buffer_ms}')
            print(f'chunk_ms={args.chunk_ms}')
        if args.jax_preallocate is not None:
            print(f'jax_preallocate={args.jax_preallocate}')
        if args.jax_mem_fraction is not None:
            print(f'jax_mem_fraction={args.jax_mem_fraction}')
        if args.jax_allocator:
            print(f'jax_allocator={args.jax_allocator}')
        if args.xla_flags_append:
            print(f'xla_flags_append={" ".join(args.xla_flags_append)}')
        print(f'batch_size={args.batch_size}')
        print(f'baseline={baseline["label"]}')
        print_summary(results, baseline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())