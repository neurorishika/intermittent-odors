#!/bin/bash
sbatch -N50 <<EOT
#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=odor-comparison
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

# order = network odor trial
srun -lN1 -r0 init_pert_trial.sh $1 '59428' '1' &
srun -lN1 -r1 init_pert_trial.sh $1 '59428' '2' &
srun -lN1 -r2 init_pert_trial.sh $1 '59428' '3' &
srun -lN1 -r3 init_pert_trial.sh $1 '59428' '4' &
srun -lN1 -r4 init_pert_trial.sh $1 '59428' '5' &
srun -lN1 -r5 init_pert_trial.sh $1 '59428' '6' &
srun -lN1 -r6 init_pert_trial.sh $1 '59428' '7' &
srun -lN1 -r7 init_pert_trial.sh $1 '59428' '8' &
srun -lN1 -r8 init_pert_trial.sh $1 '59428' '9' &
srun -lN1 -r9 init_pert_trial.sh $1 '59428' '10' &
srun -lN1 -r10 init_pert_trial.sh $1 '13674' '1' &
srun -lN1 -r11 init_pert_trial.sh $1 '13674' '2' &
srun -lN1 -r12 init_pert_trial.sh $1 '13674' '3' &
srun -lN1 -r13 init_pert_trial.sh $1 '13674' '4' &
srun -lN1 -r14 init_pert_trial.sh $1 '13674' '5' &
srun -lN1 -r15 init_pert_trial.sh $1 '13674' '6' &
srun -lN1 -r16 init_pert_trial.sh $1 '13674' '7' &
srun -lN1 -r17 init_pert_trial.sh $1 '13674' '8' &
srun -lN1 -r18 init_pert_trial.sh $1 '13674' '9' &
srun -lN1 -r19 init_pert_trial.sh $1 '13674' '10' &
srun -lN1 -r20 init_pert_trial.sh $1 '84932' '1' &
srun -lN1 -r21 init_pert_trial.sh $1 '84932' '2' &
srun -lN1 -r22 init_pert_trial.sh $1 '84932' '3' &
srun -lN1 -r23 init_pert_trial.sh $1 '84932' '4' &
srun -lN1 -r24 init_pert_trial.sh $1 '84932' '5' &
srun -lN1 -r25 init_pert_trial.sh $1 '84932' '6' &
srun -lN1 -r26 init_pert_trial.sh $1 '84932' '7' &
srun -lN1 -r27 init_pert_trial.sh $1 '84932' '8' &
srun -lN1 -r28 init_pert_trial.sh $1 '84932' '9' &
srun -lN1 -r29 init_pert_trial.sh $1 '84932' '10' &
srun -lN1 -r30 init_pert_trial.sh $1 '72957' '1' &
srun -lN1 -r31 init_pert_trial.sh $1 '72957' '2' &
srun -lN1 -r32 init_pert_trial.sh $1 '72957' '3' &
srun -lN1 -r33 init_pert_trial.sh $1 '72957' '4' &
srun -lN1 -r34 init_pert_trial.sh $1 '72957' '5' &
srun -lN1 -r35 init_pert_trial.sh $1 '72957' '6' &
srun -lN1 -r36 init_pert_trial.sh $1 '72957' '7' &
srun -lN1 -r37 init_pert_trial.sh $1 '72957' '8' &
srun -lN1 -r38 init_pert_trial.sh $1 '72957' '9' &
srun -lN1 -r39 init_pert_trial.sh $1 '72957' '10' &
srun -lN1 -r40 init_pert_trial.sh $1 '85036' '1' &
srun -lN1 -r41 init_pert_trial.sh $1 '85036' '2' &
srun -lN1 -r42 init_pert_trial.sh $1 '85036' '3' &
srun -lN1 -r43 init_pert_trial.sh $1 '85036' '4' &
srun -lN1 -r44 init_pert_trial.sh $1 '85036' '5' &
srun -lN1 -r45 init_pert_trial.sh $1 '85036' '6' &
srun -lN1 -r46 init_pert_trial.sh $1 '85036' '7' &
srun -lN1 -r47 init_pert_trial.sh $1 '85036' '8' &
srun -lN1 -r48 init_pert_trial.sh $1 '85036' '9' &
srun -lN1 -r49 init_pert_trial.sh $1 '85036' '10' &
wait
EOT
