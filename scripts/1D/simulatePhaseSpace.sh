#!/bin/bash
#SBATCH -p C-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

#SBATCH --array=0-53        # 10 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -t 0-2:00

# ---- CONFIG ----
SECONDS=0
RSIZE=15
R0SIZE=18
TOTAL_LINES=$(( RSIZE * R0SIZE ))
NJOBS=54
LINES_PER_JOB=$(( TOTAL_LINES / NJOBS))
START=$(( SLURM_ARRAY_TASK_ID * LINES_PER_JOB ))
END=$(( START + LINES_PER_JOB ))

# ---- RUN THE ASSIGNED LINES ----
i=0
while IFS= read -r line; do
    if (( i >= START && i < END )); then
        # Extract parameters from line
        read r R0 <<< "$line"

        echo "Job $SLURM_ARRAY_TASK_ID running parameters: r=$r, R0=$R0"

        # Run the simulation
        julia ~/coevolution/scripts/1D/simulateWaveRunsCluster.jl \
              "$r" "$R0" 0.2 "Normal(0, 2)" 100000 1000 10
    fi
    ((i++))
done < /home/zayas-orihuela/coevolution/scripts/1D/params.txt

echo ${SECONDS}