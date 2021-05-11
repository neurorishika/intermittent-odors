#!/bin/bash
sbatch -N10 <<EOT
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=pn-pert-trial
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

# order = network odor trial
srun -lN1 -r0 init_pert_trial.sh $1 '72957' '1' &
srun -lN1 -r1 init_pert_trial.sh $1 '72957' '2' &
srun -lN1 -r2 init_pert_trial.sh $1 '72957' '3' &
srun -lN1 -r3 init_pert_trial.sh $1 '72957' '4' &
srun -lN1 -r4 init_pert_trial.sh $1 '72957' '5' &
srun -lN1 -r5 init_pert_trial.sh $1 '72957' '6' &
srun -lN1 -r6 init_pert_trial.sh $1 '72957' '7' &
srun -lN1 -r7 init_pert_trial.sh $1 '72957' '8' &
srun -lN1 -r8 init_pert_trial.sh $1 '72957' '9' &
srun -lN1 -r9 init_pert_trial.sh $1 '72957' '10' &
wait
EOT
