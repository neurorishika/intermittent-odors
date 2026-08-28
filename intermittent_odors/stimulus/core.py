"""Shared data container and private helpers for stimulus builders."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

@dataclass
class StimulusData:
    """Container for a fully built stimulus.

    Attributes
    ----------
    current_input : np.ndarray, shape (n_n, T)
    state_vector  : np.ndarray, flat initial state
    times         : np.ndarray, shape (T,)
    time_batches  : tuple of np.ndarray
    """

    current_input: np.ndarray
    state_vector: np.ndarray
    times: np.ndarray
    time_batches: tuple



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_by_indegree(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    indegree = np.sum(matrix, axis=1)
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values, indegree, out=out, where=indegree != 0)
    return out


def _default_pnln_state(
    p_n: int,
    l_n: int,
    n_syn_ach: int,
    n_syn_fgaba: int,
    n_syn_sgaba: int,
    sim_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a standard initial state vector matching the slurm pipeline layout."""
    n_n = p_n + l_n
    sv = np.array(
        [-45.0] * p_n
        + [-45.0] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4e-4] * l_n
        + [0.0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1.0)] * n_n,
        dtype=np.float64,
    )
    sv[:n_n] += rng.normal(scale=0.75, size=n_n)
    gate_start = n_n
    gate_stop  = 2 * n_n + 4 * p_n + 3 * l_n
    sv[gate_start:gate_stop] = np.clip(
        sv[gate_start:gate_stop] + 0.04 * rng.normal(size=gate_stop - gate_start), 0.0, 1.0
    )
    if l_n:
        ca_start = gate_stop
        ca_stop  = ca_start + l_n
        sv[ca_start:ca_stop] = np.clip(
            sv[ca_start:ca_stop] * (1.0 + 0.03 * rng.normal(size=l_n)), 1e-6, None
        )
    return sv


def _split_times(times: np.ndarray, n_batches: int) -> tuple:
    time_batches = list(np.array_split(times, n_batches))
    for i in range(1, len(time_batches)):
        time_batches[i] = np.append(time_batches[i - 1][-1], time_batches[i])
    return tuple(time_batches)


