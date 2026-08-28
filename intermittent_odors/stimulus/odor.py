"""Intermittent-odor stimulus and connectivity builders."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core import StimulusData, _normalise_by_indegree

# ---------------------------------------------------------------------------
# Private RNG helpers (used by build_odor_stimulus / build_connectivity)
# ---------------------------------------------------------------------------

def _stable_seed(*parts: int) -> int:
    seed = 0
    for index, part in enumerate(parts, start=1):
        seed = (seed * 1000003 + int(part) + 97 * index) % (2 ** 32 - 1)
    return seed


def _make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset):
    if deterministic_staging:
        return np.random.RandomState(_stable_seed(graph_no, odor_seed, trial_seed, offset))
    return np.random.RandomState()

# ---------------------------------------------------------------------------
# IntermittentOdorParams
# ---------------------------------------------------------------------------


@dataclass
class IntermittentOdorParams:
    """All parameters that characterise an intermittent-odor trial.

    Defaults reproduce the published PN/LN model stimulus exactly.

    Parameters
    ----------
    n_n, p_n, l_n : int
        Total / PN / LN neuron counts.
    blocktime_ms : int
        Active odor block duration (ms).
    buffer_ms : int
        Silent buffer at start and end (ms).
    dt : float
        Simulation time step (ms).
    min_block_ms : int
        Shortest odor sub-block (determines switch granularity).
    switch_prob : float
        Probability of changing odor state at each sub-block boundary.
    active_pn_fraction : float
        Fraction of PNs that respond to this odor (paper default 9/90).
    pn_amplitude : float
        Excitatory drive amplitude onto responding PNs (paper default 0.24).
    ln_amplitude : float
        Excitatory drive amplitude onto LNs (paper default 0.0735).
    noise_fraction : float
        Multiplicative noise on the input (paper default 0.05).
    noise_floor : float
        Additive noise floor (paper default 0.001).
    batch_ms : int
        Time-batch size for iterative integration (ms).
    p_pnpn : float
        PN→PN connectivity probability (paper default 0.0).
    p_pnln : float
        PN→LN connectivity probability (paper default 0.1).
    p_lnpn : float
        LN→PN connectivity spread fraction (paper default 0.2).
    g_ach_ln : float
        Conductance amplitude for ACh (LN side, paper: 2*90*0.5*0.1*0.05).
    g_fgaba_pn : float
        Conductance amplitude for fGABA (PN side, paper: 0.3*6*1.2).
    g_fgaba_ln : float
        Conductance amplitude for fGABA (LN side, paper: 30*0.2/2*1.2).
    G_sgaba_pn : float
        Conductance amplitude for sGABA (PN side, paper: 0.3*6*0.03).
    """

    n_n: int = 120
    p_n: int = 90
    l_n: int = 30
    blocktime_ms: int = 12000
    buffer_ms: int = 500
    dt: float = 0.01
    min_block_ms: int = 50
    switch_prob: float = 0.1
    active_pn_fraction: float = 9 / 90
    pn_amplitude: float = 0.24
    ln_amplitude: float = 0.0735
    noise_fraction: float = 0.05
    noise_floor: float = 0.001
    batch_ms: int = 1000
    p_pnpn: float = 0.0
    p_pnln: float = 0.1
    p_lnpn: float = 0.2
    # Conductance amplitudes (before in-degree normalisation)
    g_ach_ln: float = 2 * 90 * 0.5 * 0.1 * 0.05
    g_fgaba_pn: float = 0.3 * 6 * 1.2
    g_fgaba_ln: float = 30 * 0.2 / 2 * 1.2
    G_sgaba_pn: float = 0.3 * 6 * 0.03

    @property
    def sim_time_ms(self) -> int:
        return self.blocktime_ms + 2 * self.buffer_ms

    @property
    def sim_res(self) -> float:
        """Alias for dt – matches ``TrialSettings.sim_res``."""
        return self.dt

    def base_conductances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (g_ach, g_fgaba, G_sgaba) population vectors (before normalisation)."""
        g_ach   = np.concatenate([np.zeros(self.p_n),       np.full(self.l_n, self.g_ach_ln)])
        g_fgaba = np.concatenate([np.full(self.p_n, self.g_fgaba_pn), np.full(self.l_n, self.g_fgaba_ln)])
        G_sgaba = np.concatenate([np.full(self.p_n, self.G_sgaba_pn), np.zeros(self.l_n)])
        return g_ach, g_fgaba, G_sgaba


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------


def build_odor_stimulus(
    params: IntermittentOdorParams,
    n_syn_ach: int,
    n_syn_fgaba: int,
    n_syn_sgaba: int,
    odor_seed: int,
    trial_seed: int,
    graph_no: int = 0,
    *,
    deterministic_staging: bool = False,
) -> StimulusData:
    """Build current_input and state_vector for an intermittent-odor trial.

    Produces **bit-for-bit identical** outputs to
    ``builders.slurm.build_current_input`` / ``build_state_vector`` when called
    with matching arguments.

    Parameters
    ----------
    params            : IntermittentOdorParams
    n_syn_{ach,fgaba,sgaba} : int  – number of non-zero entries in the matrices
    odor_seed, trial_seed, graph_no : int
    deterministic_staging : bool  – deterministic noise seeds (for testing)
    """
    # ---- Timepoints --------------------------------------------------------
    sim_time = params.blocktime_ms + 2 * params.buffer_ms
    times = np.arange(0, sim_time, params.dt)

    # ---- Split into time batches -------------------------------------------
    n_batch = max(1, int(math.ceil((times[-1] + params.dt) / params.batch_ms)))
    _batches = list(np.array_split(times, n_batch))
    for _i in range(1, len(_batches)):
        _batches[_i] = np.append(_batches[_i - 1][-1], _batches[_i])
    time_batches = tuple(np.asarray(b, dtype=np.float64) for b in _batches)

    # ---- Switch state (odor on/off pattern) --------------------------------
    active_steps = times.shape[0] - int(2 * params.buffer_ms / params.dt)
    current_input = np.ones((params.n_n, active_steps))

    switch_rng = np.random.RandomState(graph_no + odor_seed + trial_seed)
    if params.switch_prob == 0.0:
        switch_state = [1]
    else:
        switch_state = [0]
    for _val in switch_rng.choice(
        [0, 1],
        p=[1 - params.switch_prob, params.switch_prob],
        size=int(params.blocktime_ms / params.min_block_ms) - 1,
    ):
        switch_state.append(1 - switch_state[-1] if _val == 1 else switch_state[-1])
    ts = np.repeat(switch_state, int(params.min_block_ms / params.dt))

    # ---- PN / LN drive -----------------------------------------------------
    active_pn = round(params.active_pn_fraction * params.p_n)
    pn_rng = np.random.RandomState(odor_seed)
    set_pn = np.concatenate([np.ones(active_pn), np.zeros(params.p_n - active_pn)])
    pn_rng.shuffle(set_pn)

    current_input[:params.p_n, :] = params.pn_amplitude * (current_input[:params.p_n, :].T * set_pn).T * ts
    current_input[params.p_n:, :] = params.ln_amplitude * current_input[params.p_n:, :] * ts

    _buf = np.zeros((params.n_n, int(params.buffer_ms / params.dt)))
    current_input = np.concatenate([_buf, current_input, _buf], axis=1)

    noise_rng = _make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset=17)
    current_input += params.noise_fraction * current_input * noise_rng.normal(size=current_input.shape)
    current_input += params.noise_floor * noise_rng.normal(size=current_input.shape)

    # ---- Initial state vector ----------------------------------------------
    noise_rng2 = _make_noise_rng(graph_no, odor_seed, trial_seed, deterministic_staging, offset=31)
    state_vector = np.array(
        [-45] * params.p_n
        + [-45] * params.l_n
        + [0.5] * (params.n_n + 4 * params.p_n + 3 * params.l_n)
        + [2.4e-4] * params.l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * params.n_n,
        dtype=np.float64,
    )
    state_vector = state_vector + 0.005 * state_vector * noise_rng2.normal(size=state_vector.shape)

    return StimulusData(
        current_input=np.asarray(current_input, dtype=np.float64),
        state_vector=np.asarray(state_vector, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        time_batches=time_batches,
    )


def build_connectivity(
    params: IntermittentOdorParams,
    graph_no: int,
    network_dir: Any,
    *,
    normalize_conductances: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build connectivity matrices and normalised conductance vectors.

    Returns
    -------
    ach_mat, fgaba_mat, sgaba_mat : np.ndarray, shape (n_n, n_n)
    g_ach, g_fgaba, G_sgaba : np.ndarray, shape (n_n,) – normalised conductances
    """
    network_dir = Path(network_dir)

    # ---- ACh connectivity (PN→LN) ----------------------------------------
    ach_mat = np.zeros((params.n_n, params.n_n))
    ach_rng = np.random.RandomState(64163 + graph_no)
    ach_mat[params.p_n:, :params.p_n] = ach_rng.choice(
        [0.0, 1.0], size=(params.l_n, params.p_n), p=(1 - params.p_pnln, params.p_pnln),
    )
    ach_mat[:params.p_n, :params.p_n] = ach_rng.choice(
        [0.0, 1.0], size=(params.p_n, params.p_n), p=(1 - params.p_pnpn, params.p_pnpn),
    )

    # ---- fGABA / sGABA connectivity (LN→PN, LN→LN) -----------------------
    lnpn = np.zeros((params.p_n, params.l_n))
    stride = int(params.p_n / params.l_n)
    spread = (round(params.p_lnpn * params.p_n) // 2) * 2 + 1
    center = 0
    index = np.arange(params.p_n)
    for column in range(params.l_n):
        idx = index[np.arange(center - spread // 2, 1 + center + spread // 2) % params.p_n]
        lnpn[idx, column] = 1
        center += stride

    fgaba_mat = np.zeros((params.n_n, params.n_n))
    fgaba_mat[:params.p_n, params.p_n:] = lnpn
    fgaba_mat[params.p_n:, params.p_n:] = np.loadtxt(
        network_dir / f'matrix_{graph_no}.csv', delimiter=','
    )
    np.fill_diagonal(fgaba_mat, 0.0)

    sgaba_mat = np.zeros((params.n_n, params.n_n))
    sgaba_mat[:params.p_n, params.p_n:] = lnpn
    np.fill_diagonal(sgaba_mat, 0.0)

    g_ach, g_fgaba, G_sgaba = params.base_conductances()

    if normalize_conductances:
        g_ach   = _normalise_by_indegree(g_ach,   ach_mat)
        g_fgaba = _normalise_by_indegree(g_fgaba, fgaba_mat)
        G_sgaba = _normalise_by_indegree(G_sgaba, sgaba_mat)

    return (
        np.asarray(ach_mat,   dtype=np.float64),
        np.asarray(fgaba_mat, dtype=np.float64),
        np.asarray(sgaba_mat, dtype=np.float64),
        g_ach, g_fgaba, G_sgaba,
    )


def build_odor_trial(
    network,
    params: IntermittentOdorParams,
    odor_seed: int,
    trial_seed: int,
    graph_no: int = 0,
    *,
    deterministic_staging: bool = False,
    sample_stride: int | None = None,
    metadata: dict | None = None,
):
    """Assemble a complete ``ExperimentSpec`` for an intermittent-odor trial.

    Parameters
    ----------
    network : NetworkModel
        Built with ``pnln_network()`` or ``NetworkModel(...)``.
    params : IntermittentOdorParams
    odor_seed, trial_seed, graph_no : int
    deterministic_staging : bool
    sample_stride : int, optional
        Defaults to ``max(1, round(1 / params.dt))``.
    metadata : dict, optional

    Returns
    -------
    ExperimentSpec (or _CustomDynamicsExperimentSpec for custom networks)
    """
    config = network.to_config_dict()
    n_syn_ach   = int(np.sum(np.asarray(config["ach_mat"])   != 0))
    n_syn_fgaba = int(np.sum(np.asarray(config["fgaba_mat"]) != 0))
    n_syn_sgaba = int(np.sum(np.asarray(config["sgaba_mat"]) != 0))

    stim = build_odor_stimulus(
        params, n_syn_ach, n_syn_fgaba, n_syn_sgaba,
        odor_seed, trial_seed, graph_no,
        deterministic_staging=deterministic_staging,
    )

    stride = sample_stride if sample_stride is not None else max(1, round(1.0 / params.dt))

    return network.to_experiment_spec(
        stim.current_input,
        stim.state_vector,
        stim.times,
        config=config,
        input_dt=params.dt,
        sample_stride=stride,
        sample_neurons=params.n_n,
        time_batches=stim.time_batches,
        metadata={
            "family": "intermittent-odor",
            "graph_no": graph_no,
            "odor_seed": odor_seed,
            "trial_seed": trial_seed,
            **({} if metadata is None else metadata),
        },
    )


# ---------------------------------------------------------------------------
# Constant / step stimulus (for quick tests and ablation sweeps)
