#!/bin/bash
#SBATCH -p C-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

#SBATCH --array=0-17        # 10 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -t 0-48:00

# ---- CONFIG ----
SECONDS=0
MUSIZE=18
TOTAL_LINES=$(( MUSIZE ))
NJOBS=18
LINES_PER_JOB=$(( TOTAL_LINES / NJOBS))
START=$(( SLURM_ARRAY_TASK_ID * LINES_PER_JOB ))
END=$(( START + LINES_PER_JOB ))

# ---- RUN THE ASSIGNED LINES ----
i=0
while IFS= read -r line; do
    if (( i >= START && i < END )); then
        # Extract parameters from line
        read r R0 mu mutKernel Nh tmax totalRuns <<< "$line"

        echo "Job $SLURM_ARRAY_TASK_ID running parameters: r=$r, R0=$R0, mu=$mu, mutKernel=$mutKernel, Nh=$Nh, tmax=$tmax, nRuns=$totalRuns"

        # Run the simulation
        julia ~/coevolution/scripts/1D/mutantGrowth/cluster/calculateTimeSurvivalProbabilityCluster.jl \
              "$r" "$R0" "$mu" "$mutKernel" "$Nh" "$tmax" "$totalRuns"
    fi
    ((i++))
done < /home/zayas-orihuela/coevolution/scripts/1D/mutantGrowth/cluster/params.txt

echo ${SECONDS}