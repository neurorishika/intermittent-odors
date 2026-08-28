# Simulation Instructions

To run the simulations on the at different intermittencies follow the following steps:

`pnlnnetwork.py` and `single_odor_trial.py` honor `IODOR_BACKEND`. Leave it unset for the default JAX backend, or set `IODOR_BACKEND=tensorflow` to run the legacy TensorFlow reference path through the same SLURM-oriented workflow.

On the JAX backend, `single_odor_trial.py` provisions a persistent compilation cache under `slurm/__jaxcache__/` unless `IODOR_JAX_COMPILATION_CACHE_DIR` or `JAX_COMPILATION_CACHE_DIR` is already set. Override the directory explicitly if you want cache reuse across multiple working directories or nodes.

1. Set the intermittency ie. switch probability via `IODOR_SWITCH_PROB`, or change the default in `builders.py` if you want to bake a new baseline into the SLURM workflow.
2. Set the correct directory location on Line 5 of `initiate_odor_trial.sh`.
3. Clear the existing Data directory. Make sure the directory is **not** deleted.
4. Generate the simulation list by running `initialize_simulation_list.py`.
5. Run `run_simulations.py` and wait until no new jobs are being created.
6. Check for simulation completion without missing data by re-running `initialize_simulation_list.py` without any changes to the Data directory.
7. Package the data using `zip_data.sh` giving the intermittency level as the argument with "_" replacing the decimal point, eg. "0_2"
