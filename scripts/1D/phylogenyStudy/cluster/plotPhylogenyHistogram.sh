#!/bin/bash
#SBATCH -p C-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

# ---- PARAMS ----
r=19
R0=1.4
mu=0.14
nonLocalMutProbVect="[0, 5e-6]"
nonLocalJump="35"
localKernel="Normal(0,1)"
Nh=10000000
tmax=400
xmax=250
runs=200

# ---- RUN THE CODE ----
julia ~/coevolution/scripts/1D/phylogenyStudy/cluster/plotPhylogenyHistogram.jl \
        "$r" "$R0" "$mu" "$nonLocalMutProbVect" "$nonLocalJump" "$localKernel" "$Nh" "$tmax" "$xmax" "$runs"

echo ${SECONDS}