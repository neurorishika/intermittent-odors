import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from check_shared_tf_parity import build_repo_production_case

ROOT = Path(__file__).resolve().parent
FIG2_DIR = ROOT / 'fig2'
SLURM_DIR = ROOT / 'slurm'

PYTHON_BIN = ROOT / '.venv-tf312' / 'bin' / 'python'
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)


def _preserve_paths(paths):
    @contextmanager
    def manager():
        temp_dir = Path(tempfile.mkdtemp(prefix='iodor-preserve-'))
        backups = []
        try:
            for path in paths:
                if path.exists():
                    backup = temp_dir / f'{len(backups)}-{path.name}'
                    shutil.move(path, backup)
                    backups.append((path, backup))
            yield
        finally:
            for path in paths:
                if path.is_dir() and path.exists():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            for original, backup in backups:
                shutil.move(backup, original)
            shutil.rmtree(temp_dir)

    return manager()


def _run_script(cwd, script_name, args, backend, extra_env=None):
    env = os.environ.copy()
    env['IODOR_BACKEND'] = backend
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['MPLBACKEND'] = 'Agg'
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(PYTHON_BIN), script_name, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'{script_name} failed for backend={backend}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )


def _compare_arrays(name, tf_array, jax_array, atol=1e-10, rtol=1e-10):
    if tf_array.shape != jax_array.shape:
        raise SystemExit(f'{name}: shape mismatch {tf_array.shape} != {jax_array.shape}')

    matching_special = (
        (np.isnan(tf_array) & np.isnan(jax_array))
        | (np.isposinf(tf_array) & np.isposinf(jax_array))
        | (np.isneginf(tf_array) & np.isneginf(jax_array))
    )
    valid = ~matching_special
    if not np.any(valid):
        max_abs_diff = 0.0
    else:
        diff = np.abs(np.nan_to_num(tf_array[valid] - jax_array[valid], nan=np.inf, posinf=np.inf, neginf=np.inf))
        max_abs_diff = float(np.max(diff))
    print(f'{name}: max abs diff = {max_abs_diff:.3e}')

    if not np.allclose(tf_array, jax_array, atol=atol, rtol=rtol, equal_nan=True):
        raise SystemExit(f'{name}: parity check failed')


def _make_state_vector(n_n, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, seed):
    state_vector = np.array(
        [-45.0] * p_n
        + [-45.0] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4e-4] * l_n
        + [0.0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-10.0] * n_n,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    return state_vector + 0.005 * state_vector * rng.normal(size=state_vector.shape)


def _make_current_input(n_n, steps, seed):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    current_input = 0.02 * np.sin(2 * np.pi * base)[None, :] * np.ones((n_n, 1), dtype=np.float64)
    current_input += 0.03 * rng.normal(size=(n_n, steps))
    return current_input.astype(np.float64)


def _make_block_matrix(group_count, group_size):
    l_n = group_count * group_size
    matrix = np.zeros((1 + l_n, 1 + l_n), dtype=np.float64)
    for group_index in range(group_count):
        start = 1 + group_index * group_size
        stop = start + group_size
        matrix[start:stop, start:stop] = 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _write_fig2_fixed_case(metadata, current_input, state_vector, times):
    cache_dir = FIG2_DIR / '__simcache__'
    cache_dir.mkdir(exist_ok=True)
    np.save(cache_dir / 'metadata.npy', metadata, allow_pickle=True)
    np.save(cache_dir / 'current_input.npy', current_input)
    np.save(cache_dir / 'state_vector.npy', state_vector)
    np.save(cache_dir / 'time.npy', np.array([times], dtype=np.float64))


def _run_fig2_fixed_script(script_name, metadata, current_input, state_vector, times):
    cache_dir = FIG2_DIR / '__simcache__'
    output_dir = FIG2_DIR / '__simoutput__'
    output_dir.mkdir(exist_ok=True)

    protected_paths = [
        cache_dir / 'metadata.npy',
        cache_dir / 'current_input.npy',
        cache_dir / 'state_vector.npy',
        cache_dir / 'time.npy',
        output_dir / 'state_0.npy',
    ]

    with _preserve_paths(protected_paths):
        outputs = {}
        for backend in ('tensorflow', 'jax'):
            _write_fig2_fixed_case(metadata, current_input, state_vector, times)
            _run_script(FIG2_DIR, script_name, ['0'], backend)
            outputs[backend] = {
                'state': np.load(output_dir / 'state_0.npy'),
                'state_vector': np.load(cache_dir / 'state_vector.npy'),
            }
            (output_dir / 'state_0.npy').unlink()

    return outputs['tensorflow'], outputs['jax']


def _run_simple30_case():
    cache_dir = FIG2_DIR / '__simcache__'
    output_dir = FIG2_DIR / '__simoutput__'
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    graphno = 98123
    pertseed = 77119
    n_n = 7
    p_n = 1
    l_n = 6
    fgaba_mat = _make_block_matrix(group_count=2, group_size=3)
    metadata = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'fgaba_mat': fgaba_mat,
        'g_gaba': 1.5,
        'sim_res': 0.01,
    }
    times = np.arange(0.0, 2.5, 0.01, dtype=np.float64)
    current_input = _make_current_input(n_n, len(times), seed=1101)
    state_vector = _make_state_vector(n_n, p_n, l_n, 0, int(np.sum(fgaba_mat)), 0, seed=1102)

    cache_files = [
        cache_dir / f'metadata_{graphno}_{pertseed}.npy',
        cache_dir / f'current_input_{graphno}_{pertseed}.npy',
        cache_dir / f'state_vector_{graphno}_{pertseed}.npy',
        cache_dir / f'time_{graphno}_{pertseed}.npy',
        output_dir / f'state_0_{graphno}_{pertseed}.npy',
    ]

    with _preserve_paths(cache_files):
        outputs = {}
        for backend in ('tensorflow', 'jax'):
            np.save(cache_dir / f'metadata_{graphno}_{pertseed}.npy', metadata, allow_pickle=True)
            np.save(cache_dir / f'current_input_{graphno}_{pertseed}.npy', current_input)
            np.save(cache_dir / f'state_vector_{graphno}_{pertseed}.npy', state_vector)
            np.save(cache_dir / f'time_{graphno}_{pertseed}.npy', np.array([times], dtype=np.float64))
            _run_script(FIG2_DIR, 'simple30.py', ['0', str(graphno), str(pertseed)], backend)
            outputs[backend] = {
                'state': np.load(output_dir / f'state_0_{graphno}_{pertseed}.npy'),
                'state_vector': np.load(cache_dir / f'state_vector_{graphno}_{pertseed}.npy'),
            }
            (output_dir / f'state_0_{graphno}_{pertseed}.npy').unlink()

    return outputs['tensorflow'], outputs['jax']


def _run_slurm_case():
    graphno = 1
    odor_seed = 59428
    trial_seed = 991
    case_dir = SLURM_DIR / '__simcache__' / f'{graphno}_{odor_seed}_{trial_seed}'
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    output_file = case_dir / 'output_0.npy'
    try:
        config, current_input, state_vector, _, _ = build_repo_production_case(
            graph_no=graphno,
            odor_seed=odor_seed,
            trial_seed=trial_seed,
        )
        times = np.arange(0.0, 2.5, 0.01, dtype=np.float64)
        repeats = int(math.ceil(len(times) / current_input.shape[1]))
        current_input = np.tile(current_input, repeats)[:, :len(times)]

        np.save(case_dir / 'ach_mat.npy', np.asarray(config['ach_mat'], dtype=np.float64))
        np.save(case_dir / 'fgaba_mat.npy', np.asarray(config['fgaba_mat'], dtype=np.float64))
        np.save(case_dir / 'sgaba_mat.npy', np.asarray(config['sgaba_mat'], dtype=np.float64))

        outputs = {}
        for backend in ('tensorflow', 'jax'):
            np.save(case_dir / 'current_input.npy', current_input)
            np.save(case_dir / 'state_vector.npy', state_vector)
            np.save(case_dir / 'timepoint.npy', times)
            _run_script(SLURM_DIR, 'pnlnnetwork.py', [str(graphno), str(odor_seed), str(trial_seed), '0'], backend)
            outputs[backend] = {
                'state': np.load(output_file),
                'state_vector': np.load(case_dir / 'state_vector.npy'),
            }
            output_file.unlink()

        return outputs['tensorflow'], outputs['jax']
    finally:
        if case_dir.exists():
            shutil.rmtree(case_dir)


def _run_slurm_single_odor_trial_case():
    graphno = 1
    odor_seed = 59428
    trial_seed = 991
    data_file = SLURM_DIR / 'Data' / f'data_{graphno}_{odor_seed}_{trial_seed}.npy'
    case_dir = SLURM_DIR / '__simcache__' / f'{graphno}_{odor_seed}_{trial_seed}'
    test_env = {
        'IODOR_DETERMINISTIC_STAGING': '1',
        'IODOR_BLOCKTIME_MS': '200',
        'IODOR_BUFFER_MS': '50',
        'IODOR_BATCH_MS': '100',
    }

    with _preserve_paths([data_file, case_dir]):
        outputs = {}
        for backend in ('tensorflow', 'jax'):
            _run_script(
                SLURM_DIR,
                'single_odor_trial.py',
                [str(graphno), str(odor_seed), str(trial_seed)],
                backend,
                extra_env=test_env,
            )
            outputs[backend] = np.load(data_file)
            data_file.unlink()

    return outputs['tensorflow'], outputs['jax']


def main():
    n_n = 16
    p_n = 1
    l_n = 15
    fgaba_mat = _make_block_matrix(group_count=5, group_size=3)
    ach_mat = np.zeros((n_n, n_n), dtype=np.float64)
    ach_mat[1:, :1] = 1.0
    sgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
    sgaba_mat[:1, 1:] = 1.0
    np.fill_diagonal(ach_mat, 0.0)
    np.fill_diagonal(sgaba_mat, 0.0)

    times = np.arange(0.0, 2.5, 0.01, dtype=np.float64)
    current_input = _make_current_input(n_n, len(times), seed=1201)

    simple5x3_metadata = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'ach_mat': np.zeros((n_n, n_n), dtype=np.float64),
        'fgaba_mat': fgaba_mat,
        'sgaba_mat': np.zeros((n_n, n_n), dtype=np.float64),
        'g_gaba': 1.2,
        'sim_res': 0.01,
    }
    simple5x3_state = _make_state_vector(n_n, p_n, l_n, 0, int(np.sum(fgaba_mat)), 0, seed=1202)

    simple5x3pn_metadata = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'ach_mat': ach_mat,
        'fgaba_mat': fgaba_mat,
        'sgaba_mat': sgaba_mat,
        'sim_res': 0.01,
    }
    simple5x3pn_state = _make_state_vector(
        n_n,
        p_n,
        l_n,
        int(np.sum(ach_mat)),
        int(np.sum(fgaba_mat)),
        int(np.sum(sgaba_mat)),
        seed=1203,
    )

    simple30_tf, simple30_jax = _run_simple30_case()
    _compare_arrays('simple30 output', simple30_tf['state'], simple30_jax['state'])
    _compare_arrays('simple30 final state', simple30_tf['state_vector'], simple30_jax['state_vector'])

    simple5x3_tf, simple5x3_jax = _run_fig2_fixed_script(
        'simple5x3.py',
        simple5x3_metadata,
        current_input,
        simple5x3_state,
        times,
    )
    _compare_arrays('simple5x3 output', simple5x3_tf['state'], simple5x3_jax['state'])
    _compare_arrays('simple5x3 final state', simple5x3_tf['state_vector'], simple5x3_jax['state_vector'])

    simple5x3pn_tf, simple5x3pn_jax = _run_fig2_fixed_script(
        'simple5x3pn.py',
        simple5x3pn_metadata,
        current_input,
        simple5x3pn_state,
        times,
    )
    _compare_arrays('simple5x3pn output', simple5x3pn_tf['state'], simple5x3pn_jax['state'])
    _compare_arrays('simple5x3pn final state', simple5x3pn_tf['state_vector'], simple5x3pn_jax['state_vector'])

    slurm_tf, slurm_jax = _run_slurm_case()
    _compare_arrays('slurm/pnlnnetwork output', slurm_tf['state'], slurm_jax['state'])
    _compare_arrays('slurm/pnlnnetwork final state', slurm_tf['state_vector'], slurm_jax['state_vector'])

    slurm_trial_tf, slurm_trial_jax = _run_slurm_single_odor_trial_case()
    _compare_arrays('slurm/single_odor_trial output', slurm_trial_tf, slurm_trial_jax)

    print('Script-level JAX parity checks passed.')


if __name__ == '__main__':
    main()