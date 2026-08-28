# Refactor Roadmap

Tracking document for the JAX rewrite. Check items off as they land.

The refactor moved the codebase from notebook-local TensorFlow code to the layered
`model/` → `stimulus/` → `experiment` → `runtime` package. What remains is
consolidation, not new architecture.

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
| `fig2/onlyLNs.py` | Broken, needs port or removal |
| GPU parity | Not validated |

## Scope Notes

Two things are commonly assumed to be outstanding but are not:

- **Analysis notebooks need no porting.** `fig3`, `fig4`, `extended_data_fig1`,
  `supplementary_video1`, `fig5_6`, `fig7`, and `fig8` read saved `.npy` outputs
  and use only numpy, elephant, sklearn, and seaborn. They never imported the
  simulation library.
- **`fig2.ipynb` already runs on the new pipeline.** It drives simulation through
  `subprocess.call` into `simple5x3.py`, `simple5x3pn.py`, and `simple30.py`, all
  of which were ported. What remains there is duplicated stimulus construction,
  not a legacy backend dependency.

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

- [ ] Decide: port to the new pipeline, or delete both files if the figure is final
- [ ] If porting: replace the hand-built network with `pnln_network(p_n=1, l_n=30, ...)`
- [ ] If porting: replace the inline stimulus with `build_odor_stimulus(...)`
- [ ] If porting: replace the subprocess fan-out with `runner.run_time_batches(...)`
- [ ] If retiring: remove `fig2/runSimMatrix.py` as well and note it in `README.md`

## 4. Lift `fig2.ipynb` Stimulus Construction into Builders

Cells 4, 9, and 15 of `fig2/fig2.ipynb` each build `current_input` with ~30 lines of
inline numpy that `piecewise_profile` and `build_odor_stimulus` already cover.

These cells are gated behind `recalculate = False`, so a regression here stays
invisible until someone flips the flag — which argues for doing it while the
stimulus code is still fresh.

- [ ] Add the three stimulus profiles to `fig2/builders.py`
- [ ] Replace cell 4 (5×3 block stimulus) with a builder call
- [ ] Replace cell 9 (5×3 PN ramp stimulus) with a builder call
- [ ] Replace cell 15 (30-neuron perturbation stimulus) with a builder call
- [ ] Verify byte-identical `current_input` against the inline version before removing it

## 5. Validate GPU Parity

Requires `.venv-gpu-bench`. Blocks item 2 only if GPU-verified parity is wanted
before flipping the default.

- [ ] Run `check_device_parity.py` on GPU
- [ ] Record max diff for `float64` and `float32` in `docs/performance.md`
- [ ] Run `benchmark_backend_speed.py` on GPU and record the speedup

## 6. Housekeeping

- [x] Add `site/` to `.gitignore` and untrack it — mkdocs build output had been committed
      incidentally in `be5efaa`; `.gitignore` alone has no effect on already-tracked files,
      so this also required `git rm -r --cached site/` (files remain on disk)
- [x] Correct references describing `experiment` and `runtime` as subpackages; they are flat modules
- [x] Drop the stale `intermittent_odors.builders` reference from `README.md`
- [x] Drop the stale `trial_setup.py` reference from `slurm/README.md`
- [ ] Remove the stale `fig2/.ipynb_checkpoints/` and `modules/.ipynb_checkpoints/` directories if the archived notebook versions are no longer needed (they are gitignored, so this is optional)
