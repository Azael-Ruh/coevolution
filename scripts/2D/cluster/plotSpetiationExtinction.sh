#!/bin/bash
#SBATCH -p C-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

# ---- PARAMS ----
r=30
R0=2.5
muVect="[0.10, 0.12, 0.14, 0.16, 0.18, 0.20]"
nonLocalMutProb="2e-6"
nonLocalJumpVect="[0, 10, 20, 30, 40, 50, 60, 70]"
Nh=10000000
tmax=300
nCycles=30
runs=2


# ---- RUN THE CODE ----
~/.juliaup/bin/julia ~/coevolution/scripts/2D/cluster/plotSpetiationExtinction.jl \
        "$r" "$R0" "$muVect" "$nonLocalMutProb" "$nonLocalJumpVect" "$Nh" "$tmax" "$nCycles" "$runs"

echo ${SECONDS}