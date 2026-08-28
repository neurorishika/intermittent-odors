# Performance

## Backend Strategy

JAX is the default backend. TensorFlow remains available as the parity reference
and is selected with `IODOR_BACKEND=tensorflow`.

Use the backends for different goals:

| Goal | Recommended backend | Notes |
| --- | --- | --- |
| Published-model parity checks | `tensorflow` or `jax` with `IODOR_JAX_PRECISION=float64` | Default JAX precision is already `float64`. |
| Repeated exploratory runs | `jax` | Compile once and reuse. |
| Batched sweeps on CPU or GPU | `jax` | Use `run_batch(...)` or `run_time_batches_batch(...)`. |
| Custom `NetworkModel(...)` dynamics | `jax` | Custom dynamics builders are used in the JAX path. |

## Core Environment Variables

```bash
export IODOR_BACKEND=jax          # default; set only to override TensorFlow selection
export IODOR_JAX_PRECISION=float64
export JAX_PLATFORM_NAME=gpu
```

Supported JAX precision modes:

| Mode | Status |
| --- | --- |
| `float64` | **Default. The only mode safe for production runs.** |
| `float32` | Warns on selection — diverges at production rollout lengths. |
| `bfloat16` | **Refused** — produces non-finite output. Benchmarking only. |

The guards live in `backends/jax_precision.py` and are backed by the measurements
below. Set `IODOR_ALLOW_REDUCED_PRECISION=1` to override them; the benchmark and
device-parity harnesses set it themselves, since measuring those modes is their job.

```bash
# Refused outright:
IODOR_JAX_PRECISION=bfloat16 python my_script.py
# ValueError: ... produces non-finite output for this model on both CPU and GPU

# Allowed for deliberate benchmarking:
IODOR_ALLOW_REDUCED_PRECISION=1 IODOR_JAX_PRECISION=bfloat16 python my_script.py
```

## Verified CPU/GPU Parity

`check_device_parity.py` runs the same experiment under JAX on both devices and
compares the sampled trajectory and the final state. Measured on an NVIDIA RTX 6000
Ada Generation against `.venv` (CPU) and `.venv-gpu-bench` (GPU):

| Case | `float64` max abs diff | `float32` max abs diff |
| --- | --- | --- |
| `synthetic` | 5.551e-17 | 3.815e-06 |
| `repo-production` | 7.105e-15 | 3.815e-06 |
| `realistic-slurm` | 2.132e-14 | 1.144e-05 |

Reproduce with:

```bash
python check_device_parity.py                      # float64, tolerance 1e-10
python check_device_parity.py --precision float32  # float32, tolerance 1e-2
```

At `float64` the GPU agrees with the CPU to the same order as the JAX/TensorFlow
CPU parity (≤ 4.3e-14), so a GPU run is a valid substitute for a CPU run.

!!! warning "Device parity is not precision parity"
    The table above compares **the same precision across two devices**, on a short
    (200 ms) rollout. It does not say `float32` is interchangeable with `float64`,
    and it does not extend to production-length runs — see below.

## Reduced Precision Is Not Safe for This Model

Measured on the 1000 ms `realistic-slurm` benchmark, comparing each mode against the
`float64` baseline **on the same device**:

| Precision | CPU | GPU |
| --- | --- | --- |
| `float64` | baseline | baseline |
| `float32` | **diverges to non-finite** | finite, but 1.6e+01 max abs diff |
| `bfloat16` | **diverges to non-finite** | **diverges to non-finite** |

Two things follow, and neither is visible from the short-rollout parity table:

- **`bfloat16` is unusable.** It blows up on both devices despite being an accepted
  value of `IODOR_JAX_PRECISION`.
- **`float32` is unsafe for production-length rollouts.** It survives a 200 ms
  trial to ~1e-05, then goes non-finite on CPU by 1000 ms. Where it does stay finite
  (GPU), it differs from `float64` by ~16 mV — a full action potential. The model is
  spiking, so once rounding shifts one spike across a threshold the raw state arrays
  separate completely.

Use `float64`. If you have a throughput reason to consider `float32`, validate it at
your *actual* rollout length and judge it on task metrics (spike counts, rates,
timing distributions), never on `max_abs_diff`.

## Measured Throughput: Parallel Width Decides CPU vs GPU

**The device choice is set by how much parallel width the run offers — roughly
`neurons × batch` — not by either factor alone.** A single 120-neuron trial is too
narrow and the CPU wins by 3×. Widen it along *either* axis and the GPU takes over.

`jax-float64`, `realistic-slurm`, 200 ms blocktime, warm timings. Throughput is
aggregate across the batch. CPU is `.venv`; GPU is `.venv-gpu-bench` on an
NVIDIA RTX 6000 Ada Generation.

Widening along the batch axis, at the 120-neuron production network size:

| Batch | CPU sim ms/s | GPU sim ms/s | GPU advantage |
| --- | --- | --- | --- |
| 1 | 326.2 | 106.1 | 0.33× — CPU wins |
| 8 | 542.1 | 765.9 | 1.41× |
| 32 | 664.0 | 1457.0 | 2.19× |
| 64 | 745.2 | 2635.5 | **3.54×** |

Across batch 1 → 64 the GPU scales **24.8×** while the CPU scales **2.3×**. The CPU
saturates almost immediately; the GPU was still climbing at batch 64, so the ceiling
here is GPU memory, not compute.

Network size buys the same thing. At batch 1, simply making the network bigger is
enough to flip the result:

| Neurons | Batch | CPU sim ms/s | GPU sim ms/s | GPU advantage |
| --- | --- | --- | --- | --- |
| 120 | 1 | 326.2 | 106.1 | 0.33× — CPU wins |
| 480 | 1 | 32.5 | 70.2 | **2.16×** |
| 960 | 1 | 11.5 | 41.9 | **3.64×** |

So the 120-neuron production network at batch 1 sits just below the crossover on
both axes. That is the only measured configuration where the CPU is the right
choice — and it happens to be the one a single figure-generating script runs.

### Why

The integrator is a `jax.lax.scan` over time steps (`backends/jax_integrator.py`).
Time is a sequential dependency — step *n+1* needs step *n* — so **no device can
parallelize across it**. The only axes with width are neurons and batch.

At 120 neurons and batch 1, each RK step is a 120-element kernel, and the rollout is
~120,000 of them back to back. Launch overhead swamps the arithmetic and the GPU sits
mostly idle; a CPU core just walks the loop. Raising the batch multiplies the width of
every kernel without adding any steps, which is precisely the shape a GPU wants.

!!! tip "Practical rule"
    Pick the device by total parallel width, `neurons × batch`.

    - **One 120-neuron trial** (`onlyLNs.py`, a single `pnlnnetwork.py` run) → **CPU**.
      This is the only measured configuration where the CPU wins.
    - **Anything wider** — a batched sweep (`run_batch(...)`,
      `run_time_batches_batch(...)`, SLURM arrays) *or* a network of ~480+ neurons →
      **GPU**, with the largest batch that fits in memory.

Reproduce with:

```bash
python benchmark_backend_speed.py --platform cpu --case realistic-slurm \
    --batch-size 32 --skip-tensorflow
.venv-gpu-bench/bin/python benchmark_backend_speed.py --platform gpu \
    --case realistic-slurm --batch-size 32 --skip-tensorflow
```

`.venv-gpu-bench` has no TensorFlow installed, so the GPU run needs
`--skip-tensorflow`.

### JAX vs TensorFlow on CPU

The reference backend is far slower. Same case, 200 ms blocktime, batch 1:

| Backend | Warm (s) | Simulated ms/s | Speedup |
| --- | --- | --- | --- |
| `tensorflow` | 15.74 | 19.1 | 1× (reference) |
| `jax-float64` | 1.03 | 291.5 | **15.3×** |

TensorFlow remains the parity reference — it agrees with the legacy equations
exactly — but it is not a practical production backend. This is why JAX is the
default.

## Persistent Compilation Cache

The runtime honors a persistent JAX compilation cache if you set one of these:

```bash
export IODOR_JAX_COMPILATION_CACHE_DIR=$PWD/__jaxcache__
export IODOR_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export IODOR_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
```

## GPU Memory Controls

When sharing a GPU with other processes, tune preallocation explicitly.

```bash
export IODOR_JAX_PREALLOCATE=false
export IODOR_JAX_MEM_FRACTION=0.6
export IODOR_JAX_ALLOCATOR=platform
```

The SLURM-side helper `configure_runtime_environment(...)` maps these into the XLA variables that JAX reads.

## Reuse the Right Runner Method

| Method | Best use |
| --- | --- |
| `run(...)` | One full trajectory and you want every state. |
| `run_sampled(...)` | One trajectory but only sampled neurons and the final state. |
| `run_batch(...)` | Many same-shaped runs in parallel. |
| `run_time_batches(...)` | One long run split into time chunks to manage memory. |
| `run_time_batches_batch(...)` | Batched long runs with chunked integration. |

## Batched Execution Example

```python
import numpy as np

runner = experiment.compile(backend="jax")

batch_state = np.stack([experiment.state_vector for _ in range(8)], axis=0)
batch_input = np.stack([experiment.current_input for _ in range(8)], axis=0)

sampled, final_state = runner.run_sampled_batch(
    batch_state,
    batch_input,
    experiment.times,
)
```

## Time-Batched Execution Example

This is the most practical pattern for long intermittent-odor rollouts.

```python
runner = experiment.compile(backend="jax")

sampled, final_state = runner.run_time_batches(
    experiment.state_vector,
    experiment.current_input,
    progress=None,
)
```

That method uses `experiment.time_batches` by default, so the expensive setup belongs in the experiment builder, not inside the runtime loop.

## Throughput Rules That Actually Matter

- Keep repeated runs shape-stable so the same compiled graph can be reused.
- Batch across trials when the network and time shapes match.
- Prefer `run_sampled(...)` or `run_time_batches(...)` when full trajectories are not needed.
- Only lower precision after comparing against a `float64` baseline on your target metric.

## Precision Tradeoff Guidance

Recommended order:

1. Validate a representative case in `float64`.
2. Repeat it in `float32`.
3. Compare task metrics, not just raw state arrays.
4. Only then move high-volume sweeps to `float32` or `bfloat16`.

For deeper benchmarking, the repository already includes `benchmark_backend_speed.py` and the parity scripts documented in the main `README.md`.