# Project Guidelines

## Workflow

- This repository reproduces figures and analyses for the intermittent-odor paper. Prefer changes that preserve published outputs and existing data formats unless the task explicitly asks for a behavioral change.
- Start with the documented workflow in [README.md](../README.md). Use [slurm/README.md](../slurm/README.md) for the HPC simulation path instead of restating those steps in code comments or new docs.
- Most figure generation is notebook-driven. Keep notebook edits minimal and move reusable logic into Python modules when a change affects more than one notebook or script.

## Architecture

- `utils.py` contains shared statistical helpers and zip-data extraction helpers used by later figure workflows.
- `fig2/` contains smaller exploratory network simulations and duplicates much of the TensorFlow 1.x model code used elsewhere.
- `slurm/` contains the production simulation pipeline for the 120-neuron PN/LN model. It stages arrays in `__simcache__/`, runs `single_odor_trial.py`, integrates dynamics in `pnlnnetwork.py` via `tf_integrator.py`, and writes `.npy` outputs to `slurm/Data/`.
- `modules/` contains precomputed network matrices and MATLAB tooling used to generate or analyze graph structure.

## Build And Run

- The project targets an older Python stack listed in [requirements.txt](../requirements.txt), plus Jupyter for notebooks. Prefer compatibility-conscious changes over upgrading dependencies as part of unrelated work.
- There is no automated test suite or CI in this repo. If you change simulation or analysis logic, validate with the smallest reproducible script or notebook slice you can run locally and state what you did not validate.
- Run scripts from the directory they live in when they depend on relative paths. Many scripts assume local folders such as `__simcache__/`, `__datacache__/`, `../data/`, or `../modules/` exist relative to the current working directory.
- `get_data.py` is written around a Windows + 7-Zip workflow. Do not assume it works unchanged on macOS or Linux; call out cross-platform fixes explicitly if you make them.

## Conventions

- Preserve the saved-data contract unless the task explicitly allows breaking changes: simulation code writes and reads `.npy` arrays from cache directories, and analysis notebooks expect those filenames and shapes.
- Preserve reproducibility behavior unless asked otherwise. Seeds, network sizes, neuron counts, and perturbation parameters are hard-coded in several scripts and are part of how published results are reproduced.
- The main production model assumes `n_n = 120`, `p_n = 90`, and `l_n = 30`, with a flat state vector whose block ordering is shared implicitly across simulation scripts. Treat that layout as a public interface.
- The TensorFlow simulation code uses `tensorflow.compat.v1`, disabled eager execution, and often disables GPU visibility. When editing that code, keep session-based execution and dtype behavior consistent unless you are intentionally replacing the execution model.
- The SLURM scripts are cluster-specific and contain hard-coded paths, usernames, and helper-script names. Prefer parameterizing those values over silently rewriting the workflow.

## JAX Refactor Guidance

- If asked to begin a JAX migration, start with `slurm/tf_integrator.py` and the ODE definition in `slurm/pnlnnetwork.py` before touching notebooks.
- Keep the first refactor focused on parity: same state-vector layout, same output shapes, same saved-file naming, and a clear comparison path against the current TensorFlow implementation.
- Reduce duplication before broad rewrites. Shared gating-current logic appears in both `slurm/` and `fig2/`; extracting a common model module is usually a safer first improvement than rewriting every notebook.
- Safe follow-on improvements include parameterizing hard-coded paths, extracting reusable analysis helpers out of notebooks, and adding lightweight regression checks for spike detection or output-shape stability.