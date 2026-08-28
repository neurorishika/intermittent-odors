<div class="hero" markdown>

# Intermittent Odors

Design PN/LN experiments, run ablations, swap in custom connectivities, and scale the same model API from parity-first CPU runs to batched JAX execution on GPU.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[Design Experiments](designing-experiments.md){ .md-button }

</div>

> This site documents the refactored `intermittent_odors/` package.
> Use the repository `README.md` and `slurm/README.md` for figure regeneration and the legacy paper reproduction workflow.

## What This Package Gives You

<div class="card-grid" markdown>

### Standard PN/LN Experiments
Use `pnln_network(...)` with the published equations, connectivity matrices, and stimulus builders while keeping a clean programmatic API.

### Structured Ablations
Zero out projections, scale synaptic strengths, or change intrinsic conductances without dropping back to notebook-local code.

### Custom Network Designs
Build new `Population` and `Projection` graphs with the composable `NetworkModel` API when the standard PN/LN recipe is too restrictive.

### Accelerated Execution
Compile once, reuse runners, batch experiments, and tune JAX precision and memory behavior for CPU or GPU workloads.

</div>

## Recommended User Flow

1. Build or load connectivity matrices.
2. Choose either the standard `pnln_network(...)` path or a custom `NetworkModel(...)`.
3. Build a stimulus or assemble `current_input`, `times`, and `state_vector` yourself.
4. Convert that into an `ExperimentSpec`.
5. Compile a backend runner and execute with `run(...)`, `run_sampled(...)`, `run_batch(...)`, or `run_time_batches(...)`.

## API Layers

| Layer | Main module | Use when |
| --- | --- | --- |
| Network design | `intermittent_odors.model` | You are choosing neurons, channels, synapses, or connectivity. |
| Stimulus design | `intermittent_odors.stimulus` | You want intermittent odor, constant, or step-drive inputs. |
| Experiment assembly | `intermittent_odors.experiment` | You want a stable, serializable experiment object. |
| Runtime | `intermittent_odors.runtime` | You want execution, batching, sampling, or backend selection. |

## Start Here

- [Getting Started](getting-started.md) for installation, docs commands, and a minimal first run.
- [Designing Experiments](designing-experiments.md) for the standard PN/LN workflow.
- [Ablations](ablations.md) for structural and conductance sweeps.
- [Custom Networks](custom-networks.md) for the JAX-only composable network path.
- [Performance](performance.md) for backend tuning, precision tradeoffs, and batched execution.