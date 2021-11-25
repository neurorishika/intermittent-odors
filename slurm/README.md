# Simulation Instructions
To run the simulations on the at different intermittencies follow the following steps:

1. Set the intermittency ie. switch probability on Line 57 of `single_odor_trial.py`.
2. Set the correct directory location on Line 5 of `initiate_odor_trial.sh`.
3. Clear the existing Data directory. Make sure the directory is **not** deleted.
4. Generate the simulation list by running `initialize_simulation_list.py`.
5. Run `run_simulations.py` and wait until no new jobs are being created.
6. Check for simulation completion without missing data by re-running `initialize_simulation_list.py` without any changes to the Data directory.
7. Package the data using `zip_data.sh` giving the intermittency level as the argument with "_" replacing the decimal point, eg. "0_2"
