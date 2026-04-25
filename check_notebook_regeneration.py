from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SAFE_EXECUTION_ENV = {
    'IODOR_BACKEND': 'jax',
    'MPLBACKEND': 'Agg',
    'CUDA_VISIBLE_DEVICES': '-1',
    'JAX_PLATFORM_NAME': 'cpu',
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}
NUMERICAL_SUFFIXES = {'.csv', '.npy'}


@dataclass(frozen=True)
class NotebookSpec:
    path: Path
    output_paths: tuple[Path, ...]
    transient_paths: tuple[Path, ...] = ()
    force_recalculate: bool = False
    patch_python_subprocess: bool = False
    patch_ffmpeg: bool = False


NOTEBOOKS = (
    NotebookSpec(
        path=ROOT / 'fig2' / 'fig2.ipynb',
        output_paths=(ROOT / 'fig2' / 'Figures',),
        transient_paths=(
            ROOT / 'data' / '3LN',
            ROOT / 'data' / '3PN3LN',
            ROOT / 'data' / '30LN',
        ),
        force_recalculate=True,
        patch_python_subprocess=True,
    ),
    NotebookSpec(
        path=ROOT / 'fig3' / 'fig3.ipynb',
        output_paths=(ROOT / 'fig3' / 'Figures',),
    ),
    NotebookSpec(
        path=ROOT / 'fig4' / 'extended_data_fig1.ipynb',
        output_paths=(
            ROOT / 'fig4' / 'Figures' / 'SuppFig_Reliability.svg',
            ROOT / 'fig4' / 'Figures' / 'SuppFig_Trial Sequence.svg',
        ),
    ),
    NotebookSpec(
        path=ROOT / 'fig4' / 'fig4.ipynb',
        output_paths=(ROOT / 'fig4' / 'Figures', ROOT / 'fig4' / 'AnalysedData'),
        force_recalculate=True,
    ),
    NotebookSpec(
        path=ROOT / 'fig4' / 'supplementary_video1.ipynb',
        output_paths=(ROOT / 'fig4' / 'Videos', ROOT / 'fig4' / 'supplementary_video1.mp4'),
        patch_ffmpeg=True,
    ),
    NotebookSpec(
        path=ROOT / 'fig5_6' / 'fig5_6.ipynb',
        output_paths=(ROOT / 'fig5_6' / 'Figures', ROOT / 'fig5_6' / 'AnalysedData'),
        force_recalculate=True,
    ),
    NotebookSpec(
        path=ROOT / 'fig7' / 'fig7.ipynb',
        output_paths=(ROOT / 'fig7' / 'Figures', ROOT / 'fig7' / 'AnalysedData'),
        force_recalculate=True,
    ),
    NotebookSpec(
        path=ROOT / 'fig8' / 'fig8.ipynb',
        output_paths=(ROOT / 'fig8' / 'Figures', ROOT / 'fig8' / 'AnalysedData'),
        force_recalculate=True,
    ),
)


def notebook_label(spec: NotebookSpec) -> str:
    return str(spec.path.relative_to(ROOT))


def get_notebook_spec(name: str) -> NotebookSpec:
    matches = []
    for spec in NOTEBOOKS:
        relative_path = notebook_label(spec)
        if name in {relative_path, spec.path.name, spec.path.stem}:
            matches.append(spec)

    if not matches:
        available = ', '.join(notebook_label(spec) for spec in NOTEBOOKS)
        raise ValueError(f'Unknown notebook {name!r}. Available notebooks: {available}')
    if len(matches) > 1:
        labels = ', '.join(notebook_label(spec) for spec in matches)
        raise ValueError(f'Ambiguous notebook selector {name!r}. Matches: {labels}')
    return matches[0]


def select_notebooks(names: list[str] | None) -> list[NotebookSpec]:
    if not names:
        return list(NOTEBOOKS)

    selected: list[NotebookSpec] = []
    seen: set[Path] = set()
    for name in names:
        spec = get_notebook_spec(name)
        if spec.path not in seen:
            selected.append(spec)
            seen.add(spec.path)
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Regenerate notebook outputs under the JAX backend and compare against committed artifacts.')
    parser.add_argument(
        '--notebook',
        action='append',
        dest='notebooks',
        help='Notebook to verify. Accepts a relative path like fig4/fig4.ipynb, a file name, or a unique stem. Repeat to verify multiple notebooks.',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available notebook selectors and exit.',
    )
    parser.add_argument(
        '--inline',
        action='store_true',
        help='Execute selected notebooks inline instead of spawning a fresh Python process per notebook.',
    )
    parser.add_argument(
        '--numerical-only',
        action='store_true',
        help='Compare only committed numerical artifacts such as .npy and .csv outputs.',
    )
    parser.add_argument(
        '--child-notebook',
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_npy(path: Path) -> Any:
    return np.load(path, allow_pickle=True)


def load_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline='') as handle:
        return list(csv.reader(handle))


def normalize_svg_text(text: str) -> str:
    text = re.sub(r'<dc:date>.*?</dc:date>', '<dc:date></dc:date>', text, flags=re.DOTALL)

    id_mapping: dict[str, str] = {}

    def replace_generated_id(match: re.Match[str]) -> str:
        original = match.group(1)
        replacement = id_mapping.setdefault(original, f'generated-id-{len(id_mapping)}')
        return f'id="{replacement}"'

    text = re.sub(r'id="([A-Za-z_][A-Za-z0-9_.:-]*)"', replace_generated_id, text)

    for original, replacement in id_mapping.items():
        text = text.replace(f'url(#{original})', f'url(#{replacement})')
        text = text.replace(f'xlink:href="#{original}"', f'xlink:href="#{replacement}"')
        text = text.replace(f'href="#{original}"', f'href="#{replacement}"')

    return text


def compare_arrays(left: Any, right: Any, prefix: str = '') -> list[str]:
    if type(left) is not type(right):
        return [f'{prefix}type mismatch: {type(left).__name__} != {type(right).__name__}']

    if isinstance(left, np.ndarray):
        if left.dtype == object or right.dtype == object:
            if left.shape != right.shape:
                return [f'{prefix}shape mismatch: {left.shape} != {right.shape}']
            differences: list[str] = []
            for index in np.ndindex(left.shape):
                differences.extend(compare_arrays(left[index], right[index], f'{prefix}{index}: '))
                if len(differences) >= 5:
                    break
            return differences
        if left.shape != right.shape:
            return [f'{prefix}shape mismatch: {left.shape} != {right.shape}']
        if not np.issubdtype(left.dtype, np.number) or not np.issubdtype(right.dtype, np.number):
            if not np.array_equal(left, right):
                return [f'{prefix}array contents differ']
            return []
        if not np.allclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True):
            diff = np.max(np.abs(left - right))
            return [f'{prefix}max abs diff = {diff:.3e}']
        return []

    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [f'{prefix}length mismatch: {len(left)} != {len(right)}']
        differences: list[str] = []
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(compare_arrays(left_item, right_item, f'{prefix}{idx}: '))
            if len(differences) >= 5:
                break
        return differences

    if isinstance(left, (float, int, np.floating, np.integer)):
        if not np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True):
            return [f'{prefix}value mismatch: {left!r} != {right!r}']
        return []

    if left != right:
        return [f'{prefix}value mismatch: {left!r} != {right!r}']
    return []


def compare_csv_rows(left_rows: list[list[str]], right_rows: list[list[str]]) -> list[str]:
    if len(left_rows) != len(right_rows):
        return [f'row count mismatch: {len(left_rows)} != {len(right_rows)}']

    differences: list[str] = []
    for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
        if len(left_row) != len(right_row):
            differences.append(f'row {row_index}: column count mismatch: {len(left_row)} != {len(right_row)}')
            if len(differences) >= 5:
                break
            continue

        for col_index, (left_value, right_value) in enumerate(zip(left_row, right_row)):
            try:
                left_number = float(left_value)
                right_number = float(right_value)
                numeric = True
            except ValueError:
                numeric = False

            if numeric:
                if not np.isclose(left_number, right_number, atol=1e-10, rtol=1e-10, equal_nan=True):
                    differences.append(
                        f'row {row_index} col {col_index}: value mismatch: {left_number!r} != {right_number!r}'
                    )
            elif left_value != right_value:
                differences.append(
                    f'row {row_index} col {col_index}: value mismatch: {left_value!r} != {right_value!r}'
                )

            if len(differences) >= 5:
                break

        if len(differences) >= 5:
            break

    return differences


def should_compare_file(path: Path, numerical_only: bool) -> bool:
    if not numerical_only:
        return True
    return path.suffix.lower() in NUMERICAL_SUFFIXES


def compare_files(baseline: Path, current: Path) -> list[str]:
    suffix = current.suffix.lower()
    if suffix == '.npy':
        return compare_arrays(load_npy(baseline), load_npy(current))
    if suffix == '.csv':
        return compare_csv_rows(load_csv_rows(baseline), load_csv_rows(current))
    if suffix == '.svg':
        baseline_svg = normalize_svg_text(baseline.read_text())
        current_svg = normalize_svg_text(current.read_text())
        if baseline_svg != current_svg:
            return ['svg markup mismatch']
        return []
    if sha256_bytes(baseline.read_bytes()) != sha256_bytes(current.read_bytes()):
        return ['byte mismatch']
    return []


def collect_files(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    if path.is_file():
        return {path.name: path}
    files: dict[str, Path] = {}
    for child in sorted(path.rglob('*')):
        if child.is_file():
            files[str(child.relative_to(path))] = child
    return files


def compare_output_path(path: Path, backup_root: Path, numerical_only: bool = False) -> list[str]:
    relative_path = path.relative_to(ROOT)
    baseline_path = backup_root / relative_path
    baseline_files = collect_files(baseline_path)
    current_files = collect_files(path)
    differences: list[str] = []

    for file_name in sorted(set(baseline_files) | set(current_files)):
        candidate_path = path / file_name if path.is_dir() else path
        if not should_compare_file(candidate_path, numerical_only):
            continue
        baseline_file = baseline_files.get(file_name)
        current_file = current_files.get(file_name)
        display_name = str(relative_path / file_name) if path.is_dir() else str(relative_path)
        if baseline_file is None:
            differences.append(f'{display_name}: new file')
            continue
        if current_file is None:
            differences.append(f'{display_name}: missing after regeneration')
            continue
        file_differences = compare_files(baseline_file, current_file)
        if file_differences:
            differences.append(f'{display_name}: {file_differences[0]}')
    return differences


def backup_paths(paths: list[Path], backup_root: Path) -> None:
    for path in paths:
        destination = backup_root / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        elif path.exists():
            shutil.copy2(path, destination)


def restore_paths(paths: list[Path], backup_root: Path) -> None:
    for path in sorted(paths, key=lambda candidate: len(candidate.parts), reverse=True):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        backup_path = backup_root / path.relative_to(ROOT)
        if backup_path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if backup_path.is_dir():
                shutil.copytree(backup_path, path)
            else:
                shutil.copy2(backup_path, path)


@contextlib.contextmanager
def temporary_environment(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def patch_notebook_text(text: str, spec: NotebookSpec) -> str:
    patched = text
    patched = patched.replace('from_numpy_matrix', 'from_numpy_array')
    patched = patched.replace("../modules/matrix_2.csv", "../modules/networks/matrix_2.csv")
    patched = patched.replace("../modules/matrix_2_modules.csv", "../modules/networks/matrix_2_modules.csv")
    patched = patched.replace(
        '        spike_times = np.array(spike_times)\n        bst = eph.conversion.BinnedSpikeTrain(list(spike_times),bin_size=50*q.ms)',
        '        bst = eph.conversion.BinnedSpikeTrain(spike_times,bin_size=50*q.ms)',
    )
    patched = patched.replace(
        "files = os.listdir('__simoutput__/')",
        "files = [name for name in os.listdir('__simoutput__/') if name.endswith('.npy')]",
    )
    patched = patched.replace(
        "    files = [name for name in os.listdir('__simoutput__/') if name.endswith('.npy')]\n"
        "    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])\n"
        "    for i in files:\n"
        "        dataset.append(np.load(f'__simoutput__/{i}'))",
        "    files = [name for name in os.listdir('__simoutput__/') if re.fullmatch(r'state_\\d+\\.npy', name)]\n"
        "    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])\n"
        "    for i in files:\n"
        "        dataset.append(np.load(f'__simoutput__/{i}'))",
    )
    patched = patched.replace(
        "        files = [name for name in os.listdir('__simoutput__/') if name.endswith('.npy')]\n"
        "        files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])\n"
        "        for i in files:\n"
        "            dataset.append(np.load(f'__simoutput__/{i}'))",
        "        files = [name for name in os.listdir('__simoutput__/') if re.fullmatch(rf'state_\\d+_{graphno}_{pertseed}\\.npy', name)]\n"
        "        files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])\n"
        "        for i in files:\n"
        "            dataset.append(np.load(f'__simoutput__/{i}'))",
    )
    patched = patched.replace('        time.sleep(60)\n', '')
    if spec.force_recalculate:
        patched = patched.replace('recalculate = False', 'recalculate = True')
        patched = patched.replace('recalculate=False', 'recalculate=True')
    if spec.patch_python_subprocess:
        patched = patched.replace("call(['python'", "call([sys.executable")
    if spec.patch_ffmpeg:
        patched = patched.replace(
            "with zipfile.ZipFile('Videos/ffmpeg.zip', 'r') as zip_ref:\n    zip_ref.extractall('Videos/')\n\n",
            '',
        )
        patched = patched.replace("call(['ffmpeg.exe'", "call(['ffmpeg'")
        patched = patched.replace("os.remove('ffmpeg.exe')", 'pass')
        patched = patched.replace(
            "for f in filter(lambda v:\".gif\" in v, os.listdir()):\n    os.rename(f,'.archive/'+f)",
            "os.makedirs('.archive', exist_ok=True)\nfor f in filter(lambda v:\".gif\" in v, os.listdir()):\n    os.rename(f,'.archive/'+f)",
        )
    return patched


def patch_notebook(nb: nbformat.NotebookNode, spec: NotebookSpec) -> nbformat.NotebookNode:
    needs_sys_import = spec.patch_python_subprocess
    needs_shutil_import = spec.patch_ffmpeg

    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        if isinstance(source, list):
            source = ''.join(source)
        source = patch_notebook_text(source, spec)
        if needs_sys_import and 'call([sys.executable' in source and 'import sys' not in source:
            source = 'import sys\n' + source
            needs_sys_import = False
        if needs_shutil_import and "call(['ffmpeg'" in source and 'import shutil' not in source:
            source = "import shutil\nif not shutil.which('ffmpeg'):\n    raise RuntimeError('ffmpeg not found on PATH')\n" + source
            needs_shutil_import = False
        cell['source'] = source
    return nb


def execute_notebook(spec: NotebookSpec) -> None:
    for output_path in managed_paths(spec):
        if output_path.suffix:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(spec.path, as_version=4)
    nb = patch_notebook(nb, spec)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name='python3',
        resources={'metadata': {'path': str(spec.path.parent)}},
        record_timing=False,
    )
    client.execute()


def summarize_process_output(stdout: str, stderr: str, line_limit: int = 80) -> str:
    lines: list[str] = []
    if stdout.strip():
        lines.append('stdout:')
        lines.extend(stdout.strip().splitlines())
    if stderr.strip():
        lines.append('stderr:')
        lines.extend(stderr.strip().splitlines())

    if not lines:
        return 'no subprocess output captured'

    if len(lines) <= line_limit:
        return '\n'.join(lines)
    tail = '\n'.join(lines[-line_limit:])
    return f'output truncated to the last {line_limit} lines\n{tail}'


def execute_notebook_subprocess(spec: NotebookSpec) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--child-notebook', notebook_label(spec)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **SAFE_EXECUTION_ENV},
    )
    if result.returncode != 0:
        output_summary = summarize_process_output(result.stdout, result.stderr)
        raise RuntimeError(
            f'{notebook_label(spec)} subprocess execution failed with exit code {result.returncode}\n{output_summary}'
        )


def dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def managed_paths(spec: NotebookSpec) -> list[Path]:
    return dedupe_paths([*spec.output_paths, *spec.transient_paths])


def run_child_notebook(name: str) -> int:
    spec = get_notebook_spec(name)
    with temporary_environment(SAFE_EXECUTION_ENV):
        execute_notebook(spec)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for spec in NOTEBOOKS:
            print(notebook_label(spec))
        return 0

    if args.child_notebook:
        return run_child_notebook(args.child_notebook)

    selected_specs = select_notebooks(args.notebooks)
    all_output_paths = dedupe_paths([path for spec in selected_specs for path in managed_paths(spec)])
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix='iodor-notebook-regen-') as temp_dir:
        backup_root = Path(temp_dir)
        backup_paths(all_output_paths, backup_root)

        exit_code = 0
        try:
            for spec in selected_specs:
                restore_paths(managed_paths(spec), backup_root)
                print(f'Executing {notebook_label(spec)}')
                notebook_result: dict[str, Any] = {
                    'notebook': notebook_label(spec),
                    'status': 'ok',
                    'differences': [],
                }
                try:
                    if args.inline:
                        with temporary_environment(SAFE_EXECUTION_ENV):
                            execute_notebook(spec)
                    else:
                        execute_notebook_subprocess(spec)
                    differences: list[str] = []
                    for output_path in spec.output_paths:
                        differences.extend(compare_output_path(output_path, backup_root, numerical_only=args.numerical_only))
                    notebook_result['differences'] = differences
                    if differences:
                        notebook_result['status'] = 'diff'
                        exit_code = 1
                        print(f'  differences: {len(differences)}')
                        for difference in differences[:10]:
                            print(f'    - {difference}')
                    else:
                        print('  no differences detected')
                except Exception:
                    notebook_result['status'] = 'error'
                    notebook_result['error'] = traceback.format_exc()
                    exit_code = 1
                    print('  execution failed')
                    print(notebook_result['error'])
                results.append(notebook_result)
        finally:
            restore_paths(all_output_paths, backup_root)

    print('\nSummary')
    print(json.dumps(results, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))