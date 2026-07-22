#!/bin/bash
#SBATCH -p R-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

# ---- PARAMS ----
r=40
R0=1.5
muVect="[0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]"
nonLocalMutProb="1e-6"
nonLocalJumpVect="[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]"
localKernel="Normal(0,1)"
Nh=10000000
tmax=500
totalRuns=30

# ---- RUN THE ASSIGNED LINES ----
julia ~/coevolution/scripts/1D/jumpEffects/cluster/plotJumpEffectsCluster.jl \
        "$r" "$R0" "$muVect" "$nonLocalMutProb" "$nonLocalJumpVect" "$localKernel" "$Nh" "$tmax" "$totalRuns"

echo ${SECONDS}