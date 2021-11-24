#!/bin/bash
sbatch -N10 <<EOT
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=pn-pert-trial
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

# order = network odor trial
srun -lN1 -r0 initiate_odor_trial.sh $1 '13674' '1' &
srun -lN1 -r1 initiate_odor_trial.sh $1 '13674' '2' &
srun -lN1 -r2 initiate_odor_trial.sh $1 '13674' '3' &
srun -lN1 -r3 initiate_odor_trial.sh $1 '13674' '4' &
srun -lN1 -r4 initiate_odor_trial.sh $1 '13674' '5' &
srun -lN1 -r5 initiate_odor_trial.sh $1 '13674' '6' &
srun -lN1 -r6 initiate_odor_trial.sh $1 '13674' '7' &
srun -lN1 -r7 initiate_odor_trial.sh $1 '13674' '8' &
srun -lN1 -r8 initiate_odor_trial.sh $1 '13674' '9' &
srun -lN1 -r9 initiate_odor_trial.sh $1 '13674' '10' &
wait
EOT
