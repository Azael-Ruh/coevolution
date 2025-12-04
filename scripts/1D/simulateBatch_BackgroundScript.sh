#!/bin/bash
#
#SBATCH -p C-Infinite   # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH --ntasks 4 # number of tasks
#SBATCH --mem 8GB
#SBATCH -t 0-2:00 # time (D-HH:MM) 
#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR 
#SBATCH --cpus-per-task 1 # core per task

rVect=(20 80)    # rVect=(0 1 2 3 4 5 10 15 20 30 40 50 60 80 100)
R0Vect=(2 5)     # R0Vect=(1.05 1.1 1.2 1.3 1.4 1.5 1.8 2 2.4 2.8 3.5 4.2 5 6)
SECONDS=0

for r in "${rVect[@]}"
do
    for R0 in "${R0Vect[@]}"
    do
        srun -n1 -N1 --exclusive --ntasks=1 --mem-per-cpu=2GB \
            julia ~/coevolution/scripts/1D/simulateWaveCluster.jl $r $R0 0.2 "Normal(0, 2)" 100000 500 &
    done
done

wait
echo ${SECONDS}