# Getting Started

## Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

This installs both backends. JAX is the default runtime backend and its pins are
included from `requirements.txt`; TensorFlow is installed alongside it as the
parity reference backend.

## Install Documentation Dependencies

```bash
pip install -r requirements-docs.txt
```

Serve the docs locally:

```bash
mkdocs serve
```

Build a static site:

```bash
mkdocs build --strict
```

## Choose a Backend

JAX is the default backend. Set `IODOR_BACKEND` only when you want to override it.

=== "JAX (default)"

    ```bash
    export JAX_PLATFORM_NAME=cpu
    ```

    Keep `IODOR_JAX_PRECISION=float64` if you want parity-oriented runs.
    Setting `IODOR_BACKEND=jax` explicitly is equivalent to leaving it unset.

=== "TensorFlow (reference)"

    ```bash
    export IODOR_BACKEND=tensorflow
    ```

    This path preserves the legacy TensorFlow 1.x style runtime behavior and is
    the reference used by the parity checks.

## Mental Model

The package is easiest to use if you keep four objects in mind:

- `NetworkModel`: neurons, channels, synapses, and connectivity.
- `StimulusData`: `current_input`, `state_vector`, `times`, and optional `time_batches`.
- `ExperimentSpec`: stable assembly of network plus stimulus.
- `CompiledExperimentRunner`: compiled backend runner with execution methods.

## Minimal First Run

This example avoids repository-specific data files and builds a tiny synthetic PN/LN experiment from scratch.

```python
import numpy as np

from intermittent_odors.model import pnln_network
from intermittent_odors.runtime import compile_experiment
from intermittent_odors.stimulus import ConstantStimulusParams, build_constant_trial

p_n = 4
l_n = 2
n_n = p_n + l_n

ach_mat = np.zeros((n_n, n_n), dtype=np.float64)
ach_mat[p_n:, :p_n] = 1.0

fgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
fgaba_mat[:p_n, p_n:] = 1.0

sgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)

g_ach = np.concatenate([np.zeros(p_n), np.full(l_n, 0.225)])
g_fgaba = np.concatenate([np.full(p_n, 2.16), np.full(l_n, 3.6)])
G_sgaba = np.zeros(n_n, dtype=np.float64)

network = pnln_network(
    p_n=p_n,
    l_n=l_n,
    ach_mat=ach_mat,
    fgaba_mat=fgaba_mat,
    sgaba_mat=sgaba_mat,
    g_ach=g_ach,
    g_fgaba=g_fgaba,
    G_sgaba=G_sgaba,
    normalize_conductances=False,
)

stimulus = ConstantStimulusParams(
    duration_ms=20.0,
    dt=0.01,
    batch_ms=20.0,
    active_pn_fraction=0.5,
)

experiment = build_constant_trial(network, stimulus, seed=7)
runner = compile_experiment(experiment, backend="jax")
trajectory = runner.run(
    experiment.state_vector,
    experiment.current_input,
    experiment.times,
)

print(trajectory.shape)
```

## Important Timestep Rule

If you change the stimulus timestep away from the paper default `0.01 ms`, set the network `input_scale` to `1.0 / dt` so input indexing stays aligned with simulation time.

Example:

```python
params_dt = 0.05

network = pnln_network(
    p_n=p_n,
    l_n=l_n,
    ach_mat=ach_mat,
    fgaba_mat=fgaba_mat,
    sgaba_mat=sgaba_mat,
    g_ach=g_ach,
    g_fgaba=g_fgaba,
    G_sgaba=G_sgaba,
    normalize_conductances=False,
    input_scale=1.0 / params_dt,
)
```

## When To Leave the Standard Path

Stay with `pnln_network(...)` when you want the published PN/LN equations and a predictable parity story.

Move to `NetworkModel(...)` when you need one of the following:

- More than two neuron populations.
- Different channel sets across multiple PN or LN subtypes.
- Multiple projections of the same synapse family between different subpopulations.
- Connectivity layouts that do not fit the standard PN-first, LN-second ordering.