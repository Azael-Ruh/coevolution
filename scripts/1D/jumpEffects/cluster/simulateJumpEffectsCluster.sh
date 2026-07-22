#!/bin/bash
#SBATCH -p R-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

#SBATCH --array=0-59        # 10 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -t 0-48:00

# ---- CONFIG ----
SECONDS=0
MUSIZE=18
DELTASIZE=10
TOTAL_LINES=$(( MUSIZE * DELTASIZE))
NJOBS=60
LINES_PER_JOB=$(( TOTAL_LINES / NJOBS))
START=$(( SLURM_ARRAY_TASK_ID * LINES_PER_JOB ))
END=$(( START + LINES_PER_JOB ))

# ---- RUN THE ASSIGNED LINES ----
i=0
while IFS= read -r line; do
    if (( i >= START && i < END )); then
        # Extract parameters from line
        read r R0 mu mutKernel localKernel Nh tmax totalRuns <<< "$line"

        echo "Job $SLURM_ARRAY_TASK_ID running parameters: r=$r, R0=$R0, mu=$mu, mutKernel=$mutKernel, localKernel=$localKernel Nh=$Nh, tmax=$tmax, nRuns=$totalRuns"

        # Run the simulation
        julia ~/coevolution/scripts/1D/jumpEffects/cluster/calculateJumpEffectsCluster.jl \
              "$r" "$R0" "$mu" "$mutKernel" "$localKernel" "$Nh" "$tmax" "$totalRuns"
    fi
    ((i++))
done < /home/zayas-orihuela/coevolution/scripts/1D/jumpEffects/cluster/params.txt

echo ${SECONDS}