# ── Tier 1: Network description ─────────────────────────────────────────────
# Build neuron populations, channels, synapses, and projections.
# ── Submodules ───────────────────────────────────────────────────────────────
# importable as ``intermittent_odors.backends``
from intermittent_odors import backends  # noqa: F401
# ── Tier 3: Experiment assembly and execution ────────────────────────────────
# Combine a network description and a stimulus into a runnable experiment,
# then compile and integrate it in one or more time batches.
from intermittent_odors.experiment import (ExperimentSpec, NetworkSpec,
                                           PreparedExperiment, StimulusSpec,
                                           build_experiment_spec,
                                           build_network_spec,
                                           build_network_spec_from_config,
                                           build_stimulus_spec,
                                           ensure_prepared_experiment,
                                           prepare_experiment)
from intermittent_odors.model import (  # channel factory helpers; synapse factory helpers; pre-built network recipe
    A_channel, Ca_channel, ChannelSpec, K_channel, KCa_channel, Na_channel,
    NetworkModel, Population, Projection, StateVectorLayout, SynapseSpec,
    ach_synapse, fgaba_synapse, pnln_network, sgaba_synapse)
from intermittent_odors.runtime import (CompiledExperimentRunner,
                                        compile_experiment, get_backend_name,
                                        get_sampled_integrator_runner,
                                        get_sampled_integrator_runner_batch,
                                        integrate_trajectory,
                                        integrate_trajectory_batch,
                                        integrate_trajectory_sampled,
                                        integrate_trajectory_sampled_batch)
# ── Tier 2: Stimulus building ────────────────────────────────────────────────
# Parametrize odor pulses, ramps, and connectivity matrices.
from intermittent_odors.stimulus import (ConstantStimulusParams,
                                         IntermittentOdorParams,
                                         StepStimulusParams, StimulusData,
                                         build_connectivity,
                                         build_constant_stimulus,
                                         build_constant_trial,
                                         build_odor_stimulus, build_odor_trial,
                                         build_step_stimulus)

__all__ = [
    # ── Tier 1: network description ──────────────────────────────────────────
    'ChannelSpec',
    'SynapseSpec',
    'Population',
    'Projection',
    'NetworkModel',
    'StateVectorLayout',
    'A_channel',
    'Ca_channel',
    'K_channel',
    'KCa_channel',
    'Na_channel',
    'ach_synapse',
    'fgaba_synapse',
    'sgaba_synapse',
    'pnln_network',
    # ── Tier 2: stimulus building ─────────────────────────────────────────────
    'ConstantStimulusParams',
    'IntermittentOdorParams',
    'StepStimulusParams',
    'StimulusData',
    'build_connectivity',
    'build_constant_stimulus',
    'build_constant_trial',
    'build_odor_stimulus',
    'build_odor_trial',
    'build_step_stimulus',
    # ── Tier 3: experiment assembly and execution ─────────────────────────────
    'NetworkSpec',
    'StimulusSpec',
    'ExperimentSpec',
    'PreparedExperiment',
    'build_network_spec',
    'build_network_spec_from_config',
    'build_stimulus_spec',
    'build_experiment_spec',
    'ensure_prepared_experiment',
    'prepare_experiment',
    'CompiledExperimentRunner',
    'compile_experiment',
    'get_backend_name',
    'get_sampled_integrator_runner',
    'get_sampled_integrator_runner_batch',
    'integrate_trajectory',
    'integrate_trajectory_batch',
    'integrate_trajectory_sampled',
    'integrate_trajectory_sampled_batch',
    # ── Submodules ────────────────────────────────────────────────────────────
    'backends',
]
