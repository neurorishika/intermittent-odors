# Refactor Roadmap

Tracking document for the JAX rewrite. Check items off as they land.

The refactor moved the codebase from notebook-local TensorFlow code to the layered
`model/` → `stimulus/` → `experiment` → `runtime` package.

!!! success "All six items are complete"
    Every simulation entry point now goes through `ExperimentSpec` on the JAX
    backend, and parity is verified on CPU, on GPU, and against the original
    inline notebook code. The one remaining unchecked box is an optional cleanup
    of `.ipynb_checkpoints/` directories, left to the maintainer's judgement.

## Status Summary

| Area | State |
| --- | --- |
| Core package layout | Done |
| `model/`, `stimulus/`, `experiment`, `runtime`, `backends/` | Done |
| Production scripts (`single_odor_trial`, `simple5x3`, `simple5x3pn`, `simple30`, `single_pert_trial`) | Done |
| Builder decentralization (`fig2/builders.py`, `slurm/builders.py`) | Done |
| Compatibility shims removed | Done |
| JAX/TF parity (CPU) | Done — max diff ≤ 4.3e-14 |
| Single runtime entry path | Done |
| JAX as default backend | Done |
| `fig2/onlyLNs.py` | Done — ported |
| `fig2.ipynb` stimulus construction | Done — lifted into `fig2/builders.py` |
| GPU parity | Done — `float64` ≤ 2.1e-14; GPU up to 3.6× faster once width allows |

## Scope Notes

Two things are commonly assumed to be outstanding but are not:

- **Analysis notebooks need no porting.** `fig3`, `fig4`, `extended_data_fig1`,
  `supplementary_video1`, `fig5_6`, `fig7`, and `fig8` read saved `.npy` outputs
  and use only numpy, elephant, sklearn, and seaborn. They never imported the
  simulation library.
- **`fig2.ipynb` already ran on the new pipeline.** It drives simulation through
  `subprocess.call` into `simple5x3.py`, `simple5x3pn.py`, and `simple30.py`, all
  of which were ported. What remained there was duplicated stimulus construction
  (item 4 below), not a legacy backend dependency.

## 1. Collapse the Two Runtime Entry Paths

There are currently two ways to reach the runtime:

```python
# Path A — config dict (slurm/simulation.py)
experiment = ensure_prepared_experiment(config, thresholds=thresholds, ...)
runner = compile_experiment(experiment, backend=backend)

# Path B — spec object (slurm/single_odor_trial.py)
spec = trial_case_to_experiment_spec(case, settings)
runner = spec.compile(backend=backend)
```

`ExperimentSpec` is the intended door. The config-dict form should remain only as
an internal adapter, not as something callers reach for.

- [x] Add a spec-building helper to `slurm/simulation.py` that returns an `ExperimentSpec` (`build_slurm_experiment_spec`)
- [x] Move `simulate_time_batches` and `simulate_time_batches_batch` onto the spec path
- [x] Confirm `slurm/pnlnnetwork.py` still round-trips (`check_script_jax_parity.py`)
- [x] Document `ExperimentSpec` as the single supported entry point in `docs/designing-experiments.md`
- [x] Fix `slurm/builders.py` missing `sys.path` bootstrap, which broke `single_odor_trial.py` on import

## 2. Make JAX the Default Backend

The default is still `tensorflow` in two places:

- `intermittent_odors/runtime.py` — `get_backend_name()`
- `slurm/builders.py` — `configure_runtime_environment()`

With CPU parity at 4.3e-14, JAX should be the default and TensorFlow the opt-in
reference path.

- [x] Flip the default in `get_backend_name()`
- [x] Flip the default in `configure_runtime_environment()` so JAX cache setup applies by default
- [x] Fold the JAX pins into `requirements.txt` so the default backend installs by default
- [x] Update `docs/getting-started.md` and `docs/performance.md` to describe TF as the reference backend
- [x] Update `README.md` and `slurm/README.md` backend instructions
- [x] Run the test suite under both backends to confirm nothing depended on the old default

## 3. Port or Retire `fig2/onlyLNs.py`

`fig2/onlyLNs.py` and `fig2/runSimMatrix.py` were never ported. The script builds a
`metadata` dict, an inline stimulus, and an inline state-vector layout by hand,
then fans out to `simple30.py` with a `time.sleep(60)` between repetitions.

It is also **already broken** independently of the refactor: it calls
`nx.from_numpy_matrix`, removed in networkx 3.0.

**Decision: ported, not retired.** `onlyLNs.py` is the only way to regenerate the
full `data/30LN/` dataset — 10 graphs × 5 perturbation seeds, consumed by
`fig4/supplementary_video1.ipynb` and `tests/test_equivalence.py`. `fig2.ipynb`
cell 15 covers only `graphno=2, pertseed=59428`, so deleting the script would
have left 49 of the 50 committed datasets unreproducible.

- [x] Decide: port to the new pipeline, or delete both files if the figure is final
- [x] Replace the hand-built stimulus and state vector with the `fig2/builders.py` helpers
- [x] Replace the subprocess fan-out with `runner.run_time_batches(...)`, dropping the
      `__simcache__`/`__simoutput__` round-trip and the `time.sleep(60)` between repetitions
- [x] Fix `nx.from_numpy_matrix` → `nx.from_numpy_array` (also in `fig2.ipynb` cells 8 and 13)
- [x] Redirect output from the dead `fig2/Data/` path to `data/30LN/`, matching the notebook
- [x] Give both scripts an `argparse` CLI and make `onlyLNs.py` refuse to overwrite an
      existing dataset without `--force`
- [x] Verify against the old subprocess fan-out — bit-for-bit identical

The network is still built through `build_fig2_experiment_spec(...)`, exactly as the
already-parity-checked `simple30.py` does, rather than through `pnln_network(...)`.
Switching network constructors would have changed the numerics under test at the same
time as the runner change; that swap is a separate, independently verifiable step.

## 4. Lift `fig2.ipynb` Stimulus Construction into Builders

Cells 4, 9, and 15 of `fig2/fig2.ipynb` each build `current_input` with ~30 lines of
inline numpy, duplicated a fourth time in `fig2/onlyLNs.py`.

The existing `build_odor_stimulus(...)` turned out not to fit: it generates the
Markov-switching intermittent-odor drive used by the SLURM pipeline, whereas these
cells need fixed-duration blocks with a per-block perturbation envelope. They got
their own builders in `fig2/builders.py` rather than a forced fit.

These cells are gated behind `recalculate = False`, so a regression here stays
invisible until someone flips the flag — which argues for doing it while the
stimulus code is still fresh.

- [x] Add the three stimulus profiles to `fig2/builders.py`
      (`build_block_drive_stimulus`, `build_pn_ramp_stimulus`, plus the
      `build_alternating_block_pattern` / `build_shuffled_perturbation_pattern`
      block-order helpers and `build_initial_state_vector`)
- [x] Replace cell 4 (5×3 block stimulus) with a builder call
- [x] Replace cell 9 (5×3 PN ramp stimulus) with a builder call
- [x] Replace cell 15 (30-neuron perturbation stimulus) with a builder call
- [x] Verify byte-identical `current_input` against the inline version before removing it
- [x] Lock it in with `tests/test_fig2_stimulus_builders.py`, which replays each original
      inline cell body and asserts bitwise equality

Bitwise equality — not `allclose` — is the bar, because the builders have to consume
the global numpy RNG in the same order and combine floats with the same associativity
as the code they replaced. Anything looser would let regenerated data drift silently
away from what is committed under `data/`.

## 5. Validate GPU Parity

Requires `.venv-gpu-bench`. Blocks item 2 only if GPU-verified parity is wanted
before flipping the default.

- [x] Add a `--precision` flag to `check_device_parity.py`; it hard-coded `float64`,
      so `float32` could not be measured at all
- [x] Run `check_device_parity.py` on GPU
- [x] Record max diff for `float64` and `float32` in `docs/performance.md`
- [x] Run `benchmark_backend_speed.py` on GPU and record the speedup

`float64` parity holds: ≤ 2.1e-14 across all three cases, the same order as the
existing JAX/TensorFlow CPU parity. Two results were *not* what the item anticipated
and are worth carrying forward:

- **The GPU speedup depends on batch width, not network size.** The first
  measurement (batch size 1) showed the GPU 3× *slower*, which looked like a
  finding but was an artifact of giving the device no work to parallelize over.
  Sweeping batch width reverses it:

    | Batch (120 neurons) | CPU sim ms/s | GPU sim ms/s | GPU advantage |
    | --- | --- | --- | --- |
    | 1 | 326.2 | 106.1 | 0.33× |
    | 8 | 542.1 | 765.9 | 1.41× |
    | 32 | 664.0 | 1457.0 | 2.19× |
    | 64 | 745.2 | 2635.5 | 3.54× |

    Network size does the same thing on its own — at batch 1, 480 neurons gives
    2.16× and 960 neurons 3.64×. The rollout is a `lax.scan` over time, so no device
    can parallelize across time steps; width comes only from `neurons × batch`. The
    120-neuron single trial is the one configuration where the CPU wins.
- **Reduced precision is unsafe at production rollout lengths.** `bfloat16` goes
  non-finite on both devices. `float32` agrees across devices to ~1e-05 on a 200 ms
  trial, then goes non-finite on CPU by 1000 ms; on GPU it stays finite but sits
  ~16 mV from the `float64` baseline. The short-rollout parity check alone would have
  hidden this, which is why both were run.

Recorded with the full tables in `docs/performance.md`.

## 6. Housekeeping

- [x] Add `site/` to `.gitignore` and untrack it — mkdocs build output had been committed
      incidentally in `be5efaa`; `.gitignore` alone has no effect on already-tracked files,
      so this also required `git rm -r --cached site/` (files remain on disk)
- [x] Correct references describing `experiment` and `runtime` as subpackages; they are flat modules
- [x] Drop the stale `intermittent_odors.builders` reference from `README.md`
- [x] Drop the stale `trial_setup.py` reference from `slurm/README.md`
- [ ] Remove the stale `fig2/.ipynb_checkpoints/` and `modules/.ipynb_checkpoints/` directories if the archived notebook versions are no longer needed (they are gitignored, so this is optional)
