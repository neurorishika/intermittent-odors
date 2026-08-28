# Designing Experiments

## Standard PN/LN Workflow

The standard API for paper-compatible experiments is:

1. Build or load connectivity.
2. Create a `pnln_network(...)`.
3. Choose a stimulus builder.
4. Compile and run.

## `ExperimentSpec` Is the Single Entry Point

Every path into the runtime goes through an `ExperimentSpec`. Build one with
`build_experiment_spec(...)`, with a stimulus helper such as `build_odor_trial(...)`,
or with a script-local builder (`fig2/builders.py`, `slurm/builders.py`), then compile it:

```python
runner = spec.compile(backend="jax")          # or compile_experiment(spec, backend="jax")
sampled, final_state = runner.run_time_batches(spec.state_vector, spec.current_input)
```

`spec.compile(...)` is shorthand for `compile_experiment(spec.prepare(), ...)`; both are
equivalent and either is fine.

!!! note "Legacy config dictionaries"

    `prepare_experiment(...)` and the raw config-dictionary form of
    `ensure_prepared_experiment(...)` still accept a plain model config. They are
    internal adapters kept for legacy callers — new code should build an
    `ExperimentSpec` instead, so that network metadata, thresholds, and the model
    digest stay attached to the experiment.

## Using the Repository's Legacy LN-LN Graphs

The repository already ships LN-LN graph CSVs under `modules/networks/`.

Use `build_connectivity(...)` when you want to stay close to the published pipeline while working from the refactored module API.

```python
from pathlib import Path

from intermittent_odors.model import pnln_network
from intermittent_odors.runtime import compile_experiment
from intermittent_odors.stimulus import (
    IntermittentOdorParams,
    build_connectivity,
    build_odor_trial,
)

params = IntermittentOdorParams(
    n_n=120,
    p_n=90,
    l_n=30,
    blocktime_ms=6000,
    buffer_ms=500,
    dt=0.01,
    switch_prob=0.2,
    active_pn_fraction=0.15,
    batch_ms=500,
)

ach_mat, fgaba_mat, sgaba_mat, g_ach, g_fgaba, G_sgaba = build_connectivity(
    params,
    graph_no=1,
    network_dir=Path("modules/networks"),
)

network = pnln_network(
    p_n=params.p_n,
    l_n=params.l_n,
    ach_mat=ach_mat,
    fgaba_mat=fgaba_mat,
    sgaba_mat=sgaba_mat,
    g_ach=g_ach,
    g_fgaba=g_fgaba,
    G_sgaba=G_sgaba,
    normalize_conductances=False,
    input_scale=1.0 / params.dt,
)

experiment = build_odor_trial(
    network,
    params,
    odor_seed=3,
    trial_seed=11,
    graph_no=1,
)

runner = compile_experiment(experiment, backend="jax")
sampled, final_state = runner.run_time_batches(
    experiment.state_vector,
    experiment.current_input,
)
```

## Changing Neuron Counts

The packaged `modules/networks/` graphs are aligned to the published 120-neuron topology.

If you change `n_n`, `p_n`, or `l_n`, use your own connectivity matrices or generate them procedurally.

```python
import numpy as np

from intermittent_odors.model import pnln_network
from intermittent_odors.stimulus import IntermittentOdorParams, build_odor_trial

rng = np.random.default_rng(5)

params = IntermittentOdorParams(
    n_n=60,
    p_n=45,
    l_n=15,
    dt=0.05,
    blocktime_ms=3000,
    buffer_ms=250,
    switch_prob=0.15,
)

n_n = params.n_n
p_n = params.p_n

ach_mat = np.zeros((n_n, n_n), dtype=np.float64)
ach_mat[p_n:, :p_n] = rng.binomial(1, 0.12, size=(params.l_n, params.p_n))

fgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
fgaba_mat[:p_n, p_n:] = rng.binomial(1, 0.25, size=(params.p_n, params.l_n))

sgaba_mat = fgaba_mat.copy()

g_ach, g_fgaba, G_sgaba = params.base_conductances()

network = pnln_network(
    p_n=params.p_n,
    l_n=params.l_n,
    ach_mat=ach_mat,
    fgaba_mat=fgaba_mat,
    sgaba_mat=sgaba_mat,
    g_ach=g_ach,
    g_fgaba=g_fgaba,
    G_sgaba=G_sgaba,
    normalize_conductances=True,
    g_K_pn=4.2,
    g_Ca=6.0,
    input_scale=1.0 / params.dt,
)

experiment = build_odor_trial(network, params, odor_seed=1, trial_seed=2)
```

## Choosing the Right Stimulus Builder

| Builder | Use when | Notes |
| --- | --- | --- |
| `build_odor_trial(...)` | You want intermittent odor blocks. | Best match to the paper workflow. |
| `build_constant_trial(...)` | You want quick sweeps or ablations. | Fastest way to compare network changes under fixed drive. |
| `build_step_stimulus(...)` | You want a simple onset and offset pulse. | Useful for activation timing tests. |

## Compile Reuse

Reuse a compiled runner when the model topology and static parameters stay fixed and only the dynamic inputs change.

Good reuse cases:

- New initial states with the same network.
- New `current_input` arrays with the same shape.
- Repeated trials on the same compiled experiment.

Recompile when you change:

- Any connectivity matrix.
- Intrinsic conductances or reversal potentials.
- Synapse parameter arrays.
- Custom dynamics builders.

## Direct Experiment Assembly

If you already have `config`, `current_input`, `state_vector`, `times`, and thresholds, build directly with `build_experiment_spec(...)`.

```python
from intermittent_odors.experiment import build_experiment_spec

experiment = build_experiment_spec(
    config,
    current_input,
    state_vector,
    times,
    thresholds,
    input_dt=0.01,
    sample_stride=100,
    sample_neurons=config["n_n"],
    time_batches=(times,),
    metadata={"family": "custom-study"},
)
```