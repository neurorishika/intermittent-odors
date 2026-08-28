"""Channel descriptors and builtin channel factory helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from intermittent_odors.backends.jax_network import A_prop, Ca_prop, K_prop, KCa_prop, Na_prop

# ---------------------------------------------------------------------------
# Channel descriptors
# ---------------------------------------------------------------------------

_BUILTIN_GATES: dict[str, tuple[str, ...]] = {
    "K":   ("n",),
    "Na":  ("m", "h"),
    "A":   ("m", "h"),
    "Ca":  ("m", "h"),   # also has a Ca-concentration extra state
    "KCa": ("m",),       # depends on Ca concentration produced by a Ca channel
}

_BUILTIN_PROPS: dict[str, Callable] = {
    "K":   K_prop,
    "Na":  Na_prop,
    "A":   A_prop,
    "Ca":  Ca_prop,
    "KCa": KCa_prop,
}

# Default initial values for gate variables (before noise)
_GATE_DEFAULTS: dict[str, dict[str, float]] = {
    "K":   {"n": 0.5},
    "Na":  {"m": 0.5, "h": 0.5},
    "A":   {"m": 0.5, "h": 0.5},
    "Ca":  {"m": 0.5, "h": 0.5},
    "KCa": {"m": 0.5},
}
_EXTRA_STATE_DEFAULTS: dict[str, float] = {
    "Ca": 2.4e-4,   # Ca concentration
}


@dataclass(frozen=True)
class ChannelSpec:
    """Description of a single Hodgkin-Huxley channel type.

    Parameters
    ----------
    kind : str
        ``'K'``, ``'Na'``, ``'A'``, ``'Ca'``, or ``'KCa'`` for builtins;
        any other string for a custom channel (requires *props_fn*).
    conductance : float or array-like
        Maximum conductance (can be per-neuron).
    reversal : float or array-like
        Reversal potential.
    extra_params : dict, optional
        Kinetics parameters passed to the ODE builder for custom channels.
    props_fn : callable, optional
        Kinetics function ``V -> (inf_0, tau_0, ...)`` pairs.  Required for
        custom channels; may override the builtin for standard channels.
    """

    kind: str
    conductance: Any
    reversal: Any
    extra_params: dict = field(default_factory=dict)
    props_fn: Callable | None = field(default=None, compare=False, hash=False)

    def gates(self) -> tuple[str, ...]:
        if self.kind in _BUILTIN_GATES:
            return _BUILTIN_GATES[self.kind]
        raise ValueError(
            f"Custom channel {self.kind!r}: cannot infer gate names.  "
            "Add an 'gates' key to extra_params or subclass ChannelSpec."
        )

    def has_extra_state(self) -> bool:
        """True for channels that carry an extra scalar state (e.g. Ca²⁺)."""
        return self.kind in _EXTRA_STATE_DEFAULTS

    def get_props_fn(self) -> Callable:
        if self.props_fn is not None:
            return self.props_fn
        if self.kind in _BUILTIN_PROPS:
            return _BUILTIN_PROPS[self.kind]
        raise ValueError(f"No props_fn for channel kind {self.kind!r}")

    def extra_state_default(self) -> float:
        return _EXTRA_STATE_DEFAULTS.get(self.kind, 0.0)


# ---------------------------------------------------------------------------
# Convenience channel constructors with paper-default parameters
# ---------------------------------------------------------------------------

def K_channel(conductance: float = 3.6, reversal: float = -95.0) -> ChannelSpec:
    """Hodgkin-Huxley delayed-rectifier K channel."""
    return ChannelSpec("K", conductance=conductance, reversal=reversal)


def Na_channel(conductance: float = 7.15, reversal: float = 50.0) -> ChannelSpec:
    """Hodgkin-Huxley fast Na channel."""
    return ChannelSpec("Na", conductance=conductance, reversal=reversal)


def A_channel(conductance: float = 1.43, reversal: float = -95.0) -> ChannelSpec:
    """Transient K-A channel."""
    return ChannelSpec("A", conductance=conductance, reversal=reversal)


def Ca_channel(conductance: float = 5.0, reversal: float = 140.0, *,
               A_Ca: float = 2e-4, Ca0: float = 2.4e-4, t_Ca: float = 150.0) -> ChannelSpec:
    """Voltage-gated Ca channel with Ca²⁺ accumulation dynamics."""
    return ChannelSpec(
        "Ca", conductance=conductance, reversal=reversal,
        extra_params={"A_Ca": A_Ca, "Ca0": Ca0, "t_Ca": t_Ca},
    )


def KCa_channel(conductance: float = 0.045, reversal: float = -95.0) -> ChannelSpec:
    """Ca²⁺-dependent K channel (requires a Ca channel on the same population)."""
    return ChannelSpec("KCa", conductance=conductance, reversal=reversal)

