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

- `float64`
- `float32`
- `bfloat16`

Use `float64` for parity-oriented work.
Use `float32` when throughput matters more than exact agreement with the parity baselines.

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