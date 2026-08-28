"""Sweep ``onlyLNs.py`` over the five perturbation seeds for one network graph.

Usage::

    python runSimMatrix.py <graphno> [--force]

Together with the ten graphs under ``modules/networks/`` this regenerates the
``data/30LN/`` dataset. Existing outputs are preserved unless ``--force`` is
passed through to ``onlyLNs.py``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PERTURBATION_SEEDS = [59428, 13674, 84932, 72957, 85036]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('graphno', type=int)
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite existing datasets.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for pertseed in PERTURBATION_SEEDS:
        command = [sys.executable, str(HERE / 'onlyLNs.py'), str(args.graphno), str(pertseed)]
        if args.force:
            command.append('--force')
        completed = subprocess.run(command, cwd=HERE)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
