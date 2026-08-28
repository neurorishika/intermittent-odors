"""Synapse descriptors and builtin synapse factory helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Synapse descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynapseSpec:
    """Description of a synaptic kinetic model.

    Parameters
    ----------
    name : str
        Human-readable label.
    kinetics : str
        ``'ach'``   – spike-gated (Rall/Destexhe α-function style),
        ``'fgaba'`` – voltage-gated (sigmoidal transmitter release),
        ``'sgaba'`` – G-protein coupled two-state model.
    reversal : float or array-like, shape (n_dst,)
        Synaptic reversal potential.
    conductance : array-like, shape (n_dst,)
        Max conductance normalised by in-degree.
    params : dict
        Kinetics parameters.  For ``'ach'``:  ``alp``, ``bet``, ``t_max``,
        ``t_delay``, ``A`` (peak transmitter).  For ``'fgaba'``:  ``alp``,
        ``bet``, ``V0``, ``sigma``.  For ``'sgaba'``:  ``K``, ``r1``,
        ``r2``, ``r3``, ``r4``, ``G`` (conductance amplitude).
    """

    name: str
    kinetics: str   # 'ach', 'fgaba', 'sgaba'
    reversal: Any
    conductance: Any
    params: dict = field(default_factory=dict)


# Convenience synapse constructors with paper defaults

def ach_synapse(reversal: float = 0.0, conductance: Any = 0.0, *,
                alp: float = 10.0, bet: float = 0.2,
                t_max: float = 0.3, t_delay: float = 0.0,
                A: float = 0.5) -> SynapseSpec:
    """ACh-like spike-gated excitatory synapse."""
    return SynapseSpec(
        "ach", "ach", reversal=reversal, conductance=conductance,
        params={"alp": alp, "bet": bet, "t_max": t_max, "t_delay": t_delay, "A": A},
    )


def fgaba_synapse(reversal: float = -70.0, conductance: Any = 0.0, *,
                  alp: float = 10.0, bet: float = 0.16,
                  V0: float = -20.0, sigma: float = 1.5) -> SynapseSpec:
    """Fast GABAergic voltage-gated inhibitory synapse."""
    return SynapseSpec(
        "fgaba", "fgaba", reversal=reversal, conductance=conductance,
        params={"alp": alp, "bet": bet, "V0": V0, "sigma": sigma},
    )


def sgaba_synapse(reversal: float = -95.0, conductance: Any = 0.0, *,
                  K: float = 100e-12, r1: float = 1.0, r2: float = 0.025,
                  r3: float = 0.1, r4: float = 0.06) -> SynapseSpec:
    """Slow GABA-B G-protein coupled inhibitory synapse."""
    return SynapseSpec(
        "sgaba", "sgaba", reversal=reversal, conductance=conductance,
        params={"K": K, "r1": r1, "r2": r2, "r3": r3, "r4": r4},
    )


