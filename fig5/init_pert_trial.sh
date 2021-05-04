#!/bin/sh
cd $SLURM_SUBMIT_DIR

module load python/3.7
cd /home/collins/odor-states/fig5

python single_pert_trial.py $1 $2 $3
