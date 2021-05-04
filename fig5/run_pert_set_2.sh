#!/bin/bash
sbatch -N8 <<EOT
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=pn-pert-trial
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

# order = network set replicate 59428#13674#84932#72957#85036
srun -lN1 -r0 init_pert_trial.sh $1 '13674' '4' &
srun -lN1 -r1 init_pert_trial.sh $1 '13674' '5' &
srun -lN1 -r2 init_pert_trial.sh $1 '84932' '1' &
srun -lN1 -r3 init_pert_trial.sh $1 '84932' '2' &
srun -lN1 -r4 init_pert_trial.sh $1 '84932' '3' &
srun -lN1 -r5 init_pert_trial.sh $1 '84932' '4' &
srun -lN1 -r6 init_pert_trial.sh $1 '84932' '5' &
srun -lN1 -r7 init_pert_trial.sh $1 '72957' '1' &
wait
EOT
