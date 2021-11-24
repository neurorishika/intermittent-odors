#!/bin/sh
cd $SLURM_SUBMIT_DIR

module load python/3.8
cd /home/collins/odor-states/slurm

python single_odor_trial.py $1 $2 $3