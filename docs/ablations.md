# Ablations

## What Counts as an Ablation Here

In this codebase, an ablation is usually one of four operations:

- Remove a class of edges by zeroing part or all of a connectivity matrix.
- Scale or zero a conductance profile such as `g_fgaba` or `G_sgaba`.
- Change intrinsic channel strengths like `g_K_pn`, `g_A`, or `g_Ca`.
- Restrict which neurons receive input by changing stimulus parameters.

## Structural Ablations

The simplest structural ablation is to rebuild the network with modified matrices.

```python
import numpy as np

from intermittent_odors.model import pnln_network

fgaba_ablation = np.zeros_like(fgaba_mat)

network = pnln_network(
    p_n=p_n,
    l_n=l_n,
    ach_mat=ach_mat,
    fgaba_mat=fgaba_ablation,
    sgaba_mat=sgaba_mat,
    g_ach=g_ach,
    g_fgaba=np.zeros_like(g_fgaba),
    G_sgaba=G_sgaba,
    normalize_conductances=False,
    input_scale=1.0 / params.dt,
)
```

Useful variants:

- Remove all PN-to-LN excitation by zeroing `ach_mat`.
- Remove only slow inhibition by zeroing `sgaba_mat` and `G_sgaba`.
- Remove only a subpopulation by zeroing selected rows or columns.

## Conductance Sweeps

If the structure stays fixed and you only want to change synaptic strength, scale the destination conductance arrays before you rebuild the standard network.

```python
from intermittent_odors.stimulus import ConstantStimulusParams, build_constant_trial

stimulus = ConstantStimulusParams(duration_ms=200.0, dt=0.01, batch_ms=200.0)

for fgaba_scale in (1.0, 0.5, 0.1, 0.0):
    network = pnln_network(
        p_n=p_n,
        l_n=l_n,
        ach_mat=ach_mat,
        fgaba_mat=fgaba_mat,
        sgaba_mat=sgaba_mat,
        g_ach=g_ach,
        g_fgaba=fgaba_scale * g_fgaba,
        G_sgaba=G_sgaba,
        normalize_conductances=False,
    )

    experiment = build_constant_trial(
        network,
        stimulus,
        seed=7,
        metadata={"fgaba_scale": fgaba_scale},
    )

    runner = experiment.compile(backend="jax")
    sampled, final_state = runner.run_sampled(
        experiment.state_vector,
        experiment.current_input,
        experiment.times,
    )
```

## Intrinsic Property Ablations

You can change intrinsic neuron properties without leaving the standard PN/LN equation set.

```python
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
    g_A=0.0,
    g_Ca=3.0,
    g_KCa=0.0,
)
```

Typical questions this supports:

- What happens if PN A-current is removed?
- What happens if LN calcium dynamics are weakened?
- What happens if LN spike threshold or leak parameters are shifted?

## Input Ablations

Stimulus-level ablations are often cheaper than topology changes.

Examples:

- Lower `active_pn_fraction` to restrict odor drive.
- Set `ln_amplitude=0.0` to remove direct LN excitation.
- Use `ConstantStimulusParams` to replace intermittency with tonic drive.
- Use `StepStimulusParams` to isolate onset and offset effects.

```python
from intermittent_odors.stimulus import IntermittentOdorParams

params = IntermittentOdorParams(
    active_pn_fraction=0.05,
    ln_amplitude=0.0,
    switch_prob=0.0,
)
```

## Practical Sweep Pattern

For ablation studies, separate the loop into two levels:

1. Build each network or experiment variant.
2. Run repeated seeds against that fixed variant.

That gives cleaner cache behavior and clearer metadata than rebuilding everything inside one large loop.

## Metadata Discipline

Always attach ablation labels to `metadata` so downstream analysis does not have to infer what changed.

```python
experiment = build_constant_trial(
    network,
    stimulus,
    seed=13,
    metadata={
        "study": "inhibition-ablation",
        "fgaba_scale": 0.0,
        "sgaba_scale": 1.0,
    },
)
```