"""Constant and step stimulus builders."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import StimulusData, _default_pnln_state, _split_times

# ---------------------------------------------------------------------------
# Constant / step stimulus (for quick tests and ablation sweeps)
# ---------------------------------------------------------------------------


@dataclass
class ConstantStimulusParams:
    """Parameters for a simple constant-amplitude stimulus.

    Parameters
    ----------
    duration_ms : float
        Total simulation duration (ms).
    dt : float
        Time step.
    amplitude_pn : float
        Current injection amplitude for PNs.
    amplitude_ln : float
        Current injection amplitude for LNs.
    noise_fraction : float
        Multiplicative noise on the input.
    noise_floor : float
        Additive noise floor.
    batch_ms : float
        Batch size for iterative integration.
    active_pn_fraction : float
        Fraction of PNs that receive input (randomly chosen).
    """

    duration_ms: float = 100.0
    dt: float = 0.01
    amplitude_pn: float = 0.24
    amplitude_ln: float = 0.0735
    noise_fraction: float = 0.0
    noise_floor: float = 0.0
    batch_ms: float = 100.0
    active_pn_fraction: float = 1.0


def build_constant_stimulus(
    params: ConstantStimulusParams,
    n_n: int,
    p_n: int,
    n_syn_ach: int,
    n_syn_fgaba: int,
    n_syn_sgaba: int,
    seed: int = 0,
) -> StimulusData:
    """Build a constant-amplitude stimulus (useful for quick ablation tests)."""
    rng = np.random.default_rng(int(seed))
    l_n = n_n - p_n
    times = np.arange(0.0, params.duration_ms, params.dt, dtype=np.float64)
    n_steps = times.shape[0]

    active_pn = np.zeros(p_n, dtype=np.float64)
    n_active = max(1, round(p_n * params.active_pn_fraction))
    active_pn[rng.choice(p_n, size=n_active, replace=False)] = 1.0

    current_input = np.zeros((n_n, n_steps), dtype=np.float64)
    current_input[:p_n, :] = params.amplitude_pn * active_pn[:, None]
    if l_n > 0:
        current_input[p_n:, :] = params.amplitude_ln

    if params.noise_fraction > 0 or params.noise_floor > 0:
        current_input += params.noise_fraction * current_input * rng.normal(size=current_input.shape)
        current_input += params.noise_floor * rng.normal(size=current_input.shape)
        current_input = np.clip(current_input, 0.0, None)

    sim_time = float(times[-1] + params.dt)
    state_vector = _default_pnln_state(p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time, rng)

    n_batches = max(1, math.ceil(params.duration_ms / params.batch_ms))
    time_batches = _split_times(times, n_batches)

    return StimulusData(
        current_input=current_input,
        state_vector=state_vector,
        times=times,
        time_batches=time_batches,
    )


def build_constant_trial(
    network,
    params: ConstantStimulusParams,
    seed: int = 0,
    *,
    sample_stride: int | None = None,
    metadata: dict | None = None,
):
    """Convenience wrapper: constant stimulus → ``ExperimentSpec``."""
    config = network.to_config_dict()
    n_n = int(config["n_n"])
    p_n = int(config["p_n"])
    n_syn_ach   = int(np.sum(np.asarray(config["ach_mat"])   != 0))
    n_syn_fgaba = int(np.sum(np.asarray(config["fgaba_mat"]) != 0))
    n_syn_sgaba = int(np.sum(np.asarray(config["sgaba_mat"]) != 0))

    stim = build_constant_stimulus(
        params, n_n, p_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, seed=seed
    )

    stride = sample_stride if sample_stride is not None else max(1, round(1.0 / params.dt))

    return network.to_experiment_spec(
        stim.current_input,
        stim.state_vector,
        stim.times,
        config=config,
        input_dt=params.dt,
        sample_stride=stride,
        sample_neurons=n_n,
        time_batches=stim.time_batches,
        metadata={} if metadata is None else metadata,
    )


# ---------------------------------------------------------------------------
# Step-odor stimulus (binary on/off, useful for simple activation tests)
# ---------------------------------------------------------------------------


@dataclass
class StepStimulusParams:
    """Parameters for a step (square pulse) input stimulus.

    Parameters
    ----------
    onset_ms, offset_ms : float
        Start and end of the stimulus window.
    dt : float
        Time step.
    total_ms : float
        Total simulation duration.
    amplitude_pn, amplitude_ln : float
    active_pn_fraction : float
    batch_ms : float
    """

    onset_ms: float = 10.0
    offset_ms: float = 90.0
    dt: float = 0.01
    total_ms: float = 100.0
    amplitude_pn: float = 0.24
    amplitude_ln: float = 0.0735
    active_pn_fraction: float = 1.0
    batch_ms: float = 100.0


def build_step_stimulus(
    params: StepStimulusParams,
    n_n: int,
    p_n: int,
    n_syn_ach: int,
    n_syn_fgaba: int,
    n_syn_sgaba: int,
    seed: int = 0,
) -> StimulusData:
    """Build a step-pulse stimulus."""
    rng = np.random.default_rng(int(seed))
    l_n = n_n - p_n
    times = np.arange(0.0, params.total_ms, params.dt, dtype=np.float64)
    n_steps = times.shape[0]

    on_mask = (times >= params.onset_ms) & (times < params.offset_ms)

    active_pn = np.zeros(p_n, dtype=np.float64)
    n_active = max(1, round(p_n * params.active_pn_fraction))
    active_pn[rng.choice(p_n, size=n_active, replace=False)] = 1.0

    current_input = np.zeros((n_n, n_steps), dtype=np.float64)
    current_input[:p_n, :] = params.amplitude_pn * active_pn[:, None] * on_mask[None, :]
    if l_n > 0:
        current_input[p_n:, :] = params.amplitude_ln * on_mask[None, :]

    sim_time = float(times[-1] + params.dt)
    state_vector = _default_pnln_state(p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time, rng)

    n_batches = max(1, math.ceil(params.total_ms / params.batch_ms))
    time_batches = _split_times(times, n_batches)

    return StimulusData(
        current_input=current_input,
        state_vector=state_vector,
        times=times,
        time_batches=time_batches,
    )


