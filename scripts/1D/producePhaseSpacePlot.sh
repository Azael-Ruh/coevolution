#!/bin/bash
#SBATCH -p C-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

srun julia scripts/1D/plotPhaseSpace.jl