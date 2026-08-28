# Reference

## Public Package Surface

The package is intentionally organized in layers.

## `intermittent_odors.model`

| Symbol | Purpose |
| --- | --- |
| `ChannelSpec` | Low-level channel descriptor. |
| `SynapseSpec` | Low-level synapse descriptor. |
| `Population` | Homogeneous neuron block with a shared channel set. |
| `Projection` | Directed connection with a full `(n_n, n_n)` matrix. |
| `NetworkModel` | Composable network definition and state layout generator. |
| `StateVectorLayout` | Slice map for the flat state vector. |
| `pnln_network(...)` | Standard paper-compatible PN/LN network factory. |

Convenience constructors:

- `K_channel(...)`
- `Na_channel(...)`
- `A_channel(...)`
- `Ca_channel(...)`
- `KCa_channel(...)`
- `ach_synapse(...)`
- `fgaba_synapse(...)`
- `sgaba_synapse(...)`

## `intermittent_odors.stimulus`

| Symbol | Purpose |
| --- | --- |
| `StimulusData` | Holds `current_input`, `state_vector`, `times`, and `time_batches`. |
| `IntermittentOdorParams` | Parameter bundle for intermittent odor trials. |
| `ConstantStimulusParams` | Parameter bundle for tonic-drive ablation studies. |
| `StepStimulusParams` | Parameter bundle for onset/offset pulses. |
| `build_connectivity(...)` | Load legacy connectivity and normalized conductances. |
| `build_odor_trial(...)` | Standard intermittent-odor experiment builder. |
| `build_constant_trial(...)` | Standard constant-drive experiment builder. |
| `build_step_stimulus(...)` | Step-drive stimulus builder. |

## `intermittent_odors.experiment`

| Symbol | Purpose |
| --- | --- |
| `NetworkSpec` | Frozen network payload. |
| `StimulusSpec` | Frozen stimulus payload. |
| `ExperimentSpec` | Stable experiment assembly. |
| `PreparedExperiment` | Normalized, hashable compile-time representation. |
| `prepare_experiment(...)` | Convert a legacy config plus thresholds into `PreparedExperiment`. |
| `build_experiment_spec(...)` | Build an `ExperimentSpec` directly from raw arrays. |
| `ensure_prepared_experiment(...)` | Accept config, `ExperimentSpec`, or `PreparedExperiment` and normalize it. |

## `intermittent_odors.runtime`

| Symbol | Purpose |
| --- | --- |
| `compile_experiment(...)` | Return a `CompiledExperimentRunner`. |
| `CompiledExperimentRunner.run(...)` | Full trajectory for one run. |
| `CompiledExperimentRunner.run_sampled(...)` | Sampled trajectory plus final state. |
| `CompiledExperimentRunner.run_batch(...)` | Full trajectories for a batch. |
| `CompiledExperimentRunner.run_sampled_batch(...)` | Sampled batched trajectories. |
| `CompiledExperimentRunner.run_time_batches(...)` | Chunked long-run execution. |
| `CompiledExperimentRunner.run_time_batches_batch(...)` | Chunked batched execution. |
| `get_backend_name(...)` | Resolve `jax` versus `tensorflow`. |

## Builder Utilities

Builders live next to the scripts that use them, not in the package. There is no
`intermittent_odors.builders` module.

`fig2/builders.py` — figure 2 networks, stimuli, and initial states:

| Export | Purpose |
| --- | --- |
| `build_fig2_experiment_spec(...)` | Assemble a fig2 network as an `ExperimentSpec`. |
| `build_block_drive_stimulus(...)` | Constant baseline drive with a perturbation pulse per block (LN-only networks). |
| `build_pn_ramp_stimulus(...)` | Ramped perturbation onto the PNs over a constant LN baseline. |
| `build_alternating_block_pattern(...)` | Block order in which no odor repeats back-to-back. |
| `build_shuffled_perturbation_pattern(...)` | Blocks perturbing a random half of the LNs. |
| `build_initial_state_vector(...)` | Resting state with per-trial multiplicative jitter. |
| `block_pulse_filter(...)`, `pn_ramp_filter(...)` | The per-block temporal envelopes. |
| `piecewise_profile(...)` | PN-then-LN parameter profile. |

`slurm/builders.py` — trial staging for the SLURM pipeline:

- `TrialSettings`
- `TrialCase`
- `build_trial_case(...)`
- `trial_case_to_experiment_spec(...)`
- `configure_runtime_environment(...)`

The fig2 stimulus builders reproduce the original inline notebook code bit-for-bit;
`tests/test_fig2_stimulus_builders.py` enforces that. They consume the **global**
numpy RNG, so seed with `np.random.seed(...)` and call them in the documented order
if you need to reproduce a committed dataset.

## API Constraints Worth Remembering

- `pnln_network(...)` is the parity-preserving standard model path.
- `build_constant_stimulus(...)`, `build_step_stimulus(...)`, and `build_odor_stimulus(...)` assume PN-first then LN ordering.
- Custom `NetworkModel(...)` dynamics should be run with the JAX backend.
- If `dt` changes, set `input_scale=1.0 / dt` when building the network.

## Suggested Import Style

For user-facing scripts, prefer the top-level package exports:

```python
from intermittent_odors import (
    IntermittentOdorParams,
    ConstantStimulusParams,
    build_connectivity,
    build_odor_trial,
    build_constant_trial,
    compile_experiment,
    pnln_network,
)
```

Drop to submodules when you need lower-level control, especially for custom `Population`, `Projection`, or `NetworkModel` work.