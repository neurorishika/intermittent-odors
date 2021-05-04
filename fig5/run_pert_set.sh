#!/bin/bash
sbatch -N25 <<EOT
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=odor-comparison
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

# order = network set replicate 59428#13674#84932#72957#85036
srun -lN1 -r0 init_pert_trial.sh $1 '59428' '1' &
srun -lN1 -r1 init_pert_trial.sh $1 '59428' '2' &
srun -lN1 -r2 init_pert_trial.sh $1 '59428' '3' &
srun -lN1 -r3 init_pert_trial.sh $1 '59428' '4' &
srun -lN1 -r4 init_pert_trial.sh $1 '59428' '5' &
srun -lN1 -r5 init_pert_trial.sh $1 '13674' '1' &
srun -lN1 -r6 init_pert_trial.sh $1 '13674' '2' &
srun -lN1 -r7 init_pert_trial.sh $1 '13674' '3' &
srun -lN1 -r8 init_pert_trial.sh $1 '13674' '4' &
srun -lN1 -r9 init_pert_trial.sh $1 '13674' '5' &
srun -lN1 -r10 init_pert_trial.sh $1 '84932' '1' &
srun -lN1 -r11 init_pert_trial.sh $1 '84932' '2' &
srun -lN1 -r12 init_pert_trial.sh $1 '84932' '3' &
srun -lN1 -r13 init_pert_trial.sh $1 '84932' '4' &
srun -lN1 -r14 init_pert_trial.sh $1 '84932' '5' &
srun -lN1 -r15 init_pert_trial.sh $1 '72957' '1' &
srun -lN1 -r16 init_pert_trial.sh $1 '72957' '2' &
srun -lN1 -r17 init_pert_trial.sh $1 '72957' '3' &
srun -lN1 -r18 init_pert_trial.sh $1 '72957' '4' &
srun -lN1 -r19 init_pert_trial.sh $1 '72957' '5' &
srun -lN1 -r20 init_pert_trial.sh $1 '85036' '1' &
srun -lN1 -r21 init_pert_trial.sh $1 '85036' '2' &
srun -lN1 -r22 init_pert_trial.sh $1 '85036' '3' &
srun -lN1 -r23 init_pert_trial.sh $1 '85036' '4' &
srun -lN1 -r24 init_pert_trial.sh $1 '85036' '5' &
wait
EOT
