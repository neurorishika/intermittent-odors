# Custom Networks

## When To Use `NetworkModel(...)`

Use the composable model API when `pnln_network(...)` is no longer expressive enough.

Examples:

- Multiple PN subtypes with different intrinsic conductances.
- More than one LN class.
- Extra excitatory or inhibitory projections between subpopulations.
- A state-vector layout that should be derived from explicit channel descriptors rather than implied by the legacy config format.

## Constraint: Custom Networks Are a JAX Path

Custom `NetworkModel(...)` experiments attach a custom dynamics builder that is consumed by the JAX runtime path.

Use `backend="jax"` for custom networks.

## Example: Two PN Subtypes Plus One LN Population

This example keeps the model explicit and avoids assumptions baked into the standard stimulus helpers.

```python
import numpy as np

from intermittent_odors.model import (
    Population,
    Projection,
    NetworkModel,
    K_channel,
    Na_channel,
    A_channel,
    Ca_channel,
    KCa_channel,
    ach_synapse,
    fgaba_synapse,
    sgaba_synapse,
)
from intermittent_odors.runtime import compile_experiment

pn_fast = Population(
    "PN_fast",
    4,
    channels=(K_channel(3.6), Na_channel(7.15), A_channel(1.43)),
    threshold=0.0,
)

pn_slow = Population(
    "PN_slow",
    4,
    channels=(K_channel(4.2), Na_channel(6.5), A_channel(0.7)),
    threshold=0.0,
)

ln = Population(
    "LN",
    2,
    channels=(K_channel(36.0), Ca_channel(5.0), KCa_channel(0.045)),
    E_L=-50.0,
    threshold=-20.0,
)

n_n = pn_fast.n + pn_slow.n + ln.n
pn_fast_slice = slice(0, 4)
pn_slow_slice = slice(4, 8)
ln_slice = slice(8, 10)

ach_to_ln = np.zeros((n_n, n_n), dtype=np.float64)
ach_to_ln[ln_slice, pn_fast_slice] = 1.0
ach_to_ln[ln_slice, pn_slow_slice] = 1.0

ach_fast_to_slow = np.zeros((n_n, n_n), dtype=np.float64)
ach_fast_to_slow[pn_slow_slice, pn_fast_slice] = 1.0

fgaba_to_pn = np.zeros((n_n, n_n), dtype=np.float64)
fgaba_to_pn[pn_fast_slice, ln_slice] = 1.0
fgaba_to_pn[pn_slow_slice, ln_slice] = 1.0

sgaba_to_pn = fgaba_to_pn.copy()

g_ach_to_ln = np.concatenate([np.zeros(8), np.full(2, 0.2)])
g_ach_fast_to_slow = np.concatenate([np.zeros(4), np.full(4, 0.05), np.zeros(2)])
g_fgaba_to_pn = np.concatenate([np.full(8, 1.8), np.zeros(2)])
G_sgaba_to_pn = np.concatenate([np.full(8, 0.05), np.zeros(2)])

network = NetworkModel(
    populations=[pn_fast, pn_slow, ln],
    projections=[
        Projection("PN_fast", "LN", ach_synapse(conductance=g_ach_to_ln), ach_to_ln),
        Projection("PN_fast", "PN_slow", ach_synapse(conductance=g_ach_fast_to_slow), ach_fast_to_slow),
        Projection("LN", "PN_fast", fgaba_synapse(conductance=g_fgaba_to_pn), fgaba_to_pn),
        Projection("LN", "PN_fast", sgaba_synapse(conductance=G_sgaba_to_pn), sgaba_to_pn),
    ],
    global_params={"input_scale": 20.0},
)

times = np.arange(0.0, 80.0, 0.05, dtype=np.float64)
current_input = np.zeros((n_n, times.size), dtype=np.float64)

fast_mask = (times >= 10.0) & (times < 45.0)
slow_mask = (times >= 30.0) & (times < 70.0)

current_input[pn_fast_slice, fast_mask] = 0.24
current_input[pn_slow_slice, slow_mask] = 0.18
current_input[ln_slice, fast_mask | slow_mask] = 0.05

state_vector = network.default_state_vector(
    rng_or_seed=4,
    sim_time=float(times[-1] + 0.05),
    noise_scale=0.01,
)

experiment = network.to_experiment_spec(
    current_input,
    state_vector,
    times,
    input_dt=0.05,
    sample_stride=20,
    sample_neurons=n_n,
    time_batches=(times,),
    metadata={"family": "custom-subtypes"},
)

runner = compile_experiment(experiment, backend="jax")
sampled, final_state = runner.run_sampled(state_vector, current_input, times)
```

## Why This Example Builds Inputs Manually

The convenience stimulus helpers assume a paper-style ordering where PNs occupy the first `p_n` indices and LNs occupy the trailing `l_n` indices.

That assumption is fine for `pnln_network(...)`.

For truly custom layouts, build the following directly:

- `times`
- `current_input`
- `state_vector`

Then call `network.to_experiment_spec(...)`.

## State Vector Initialization

Use `network.default_state_vector(...)` when the network layout comes from the composable model API.

It ensures gate, synapse, calcium, and fire-time sections match the generated layout.

## Debugging Tips

If a custom network fails, check these first:

- Every projection matrix has shape `(n_n, n_n)`.
- Conductance arrays line up with destination-neuron indexing.
- `input_scale` matches `1.0 / input_dt`.
- Your populations are declared in the same order you assume when writing `current_input` slices.