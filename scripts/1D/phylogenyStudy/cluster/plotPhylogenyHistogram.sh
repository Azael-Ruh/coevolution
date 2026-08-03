#!/bin/bash
#SBATCH -p R-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

# ---- PARAMS ----
r=18
R0=1.35
muVect="[0.17]"
nonLocalMutProb="[0, 5e-6]"
nonLocalJumpVect="[0, 30]"
localKernel="Normal(0,1)"
Nh=10000000
tmax=400
xmax=250
runs=50

# ---- RUN THE CODE ----
julia ~/coevolution/scripts/1D/phylogenyStudy/cluster/plotPhylogenyHistogram.jl \
        "$r" "$R0" "$muVect" "$nonLocalMutProb" "$nonLocalJumpVect" "$localKernel" "$Nh" "$tmax" "$xmax" "$runs"

echo ${SECONDS}