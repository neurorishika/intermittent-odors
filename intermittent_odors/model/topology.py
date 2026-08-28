"""Population, Projection, and StateVectorLayout descriptors."""
from __future__ import annotations

from dataclasses import dataclass

from .channels import ChannelSpec
from .synapses import SynapseSpec

# ---------------------------------------------------------------------------
# Population and Projection descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Population:
    """A homogeneous group of neurons sharing the same channel configuration.

    Parameters
    ----------
    name : str
    n : int
        Number of neurons.
    channels : tuple of ChannelSpec
        Active ion channels.  Order determines state-vector layout.
    C_m : float
        Membrane capacitance (µF/cm²).
    g_L : float
        Leak conductance (mS/cm²).
    E_L : float
        Leak reversal potential (mV).
    threshold : float
        Spike-detection threshold for fire-time tracking (mV).
        Use 0.0 for PNs (depolarising spikes) and –20.0 for LNs.
    """

    name: str
    n: int
    channels: tuple[ChannelSpec, ...]
    C_m: float = 1.0
    g_L: float = 0.3
    E_L: float = -64.0
    threshold: float = 0.0

    def channel(self, kind: str) -> ChannelSpec:
        for ch in self.channels:
            if ch.kind == kind:
                return ch
        raise KeyError(f"Population {self.name!r} has no channel {kind!r}")

    def has_channel(self, kind: str) -> bool:
        return any(ch.kind == kind for ch in self.channels)


@dataclass(frozen=True)
class Projection:
    """A directed synaptic projection between two named populations.

    Parameters
    ----------
    src : str
        Source population name.
    dst : str
        Destination population name.
    synapse : SynapseSpec
    matrix : array-like, shape (n_n, n_n)
        Full connectivity matrix (zeros for absent connections).
    """

    src: str
    dst: str
    synapse: SynapseSpec
    matrix: Any   # np.ndarray at runtime


# ---------------------------------------------------------------------------
# State-vector layout
# ---------------------------------------------------------------------------

@dataclass
class StateVectorLayout:
    """Slice info for every logical section of the flat state vector.

    Attributes
    ----------
    n_n : int
        Total neuron count.
    sections : dict[str, slice]
        Maps section name → ``slice(start, stop)`` into the state vector.
    state_size : int
        Total length of the state vector.
    """

    n_n: int
    sections: dict
    state_size: int

    def get(self, name: str) -> slice:
        return self.sections[name]

    def __repr__(self):
        lines = [f"StateVectorLayout(n_n={self.n_n}, state_size={self.state_size})"]
        for name, sl in self.sections.items():
            lines.append(f"  {name:45s}  [{sl.start}:{sl.stop}]  (size {sl.stop - sl.start})")
        return "\n".join(lines)


