"""Composable Hodgkin-Huxley network model description layer.

Provides a declarative API for building biophysical neuron network models
without hardcoding the state-vector layout.

Submodules
----------
channels   : ChannelSpec and builtin channel factory helpers.
synapses   : SynapseSpec and builtin synapse factory helpers.
topology   : Population, Projection, StateVectorLayout.
network    : NetworkModel, custom dynamics builder, pnln_network factory.
"""
from .channels import (
    ChannelSpec,
    K_channel,
    Na_channel,
    A_channel,
    Ca_channel,
    KCa_channel,
)
from .synapses import (
    SynapseSpec,
    ach_synapse,
    fgaba_synapse,
    sgaba_synapse,
)
from .topology import Population, Projection, StateVectorLayout
from .network import NetworkModel, pnln_network

__all__ = [
    # channel descriptors
    "ChannelSpec",
    "K_channel",
    "Na_channel",
    "A_channel",
    "Ca_channel",
    "KCa_channel",
    # synapse descriptors
    "SynapseSpec",
    "ach_synapse",
    "fgaba_synapse",
    "sgaba_synapse",
    # network topology
    "Population",
    "Projection",
    "StateVectorLayout",
    # network model and factory
    "NetworkModel",
    "pnln_network",
]
