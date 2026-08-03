#!/bin/bash
#SBATCH -p R-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

#SBATCH --array=0-99        # 100 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -t 0-48:00

# ---- CONFIG ----
SECONDS=0
MUSIZE=2
NONLOCPROBSIZE=2
RUNSIZE=200
TOTAL_LINES=$(( MUSIZE * NONLOCPROBSIZE * RUNSIZE))
NJOBS=100
LINES_PER_JOB=$(( TOTAL_LINES / NJOBS))
START=$(( SLURM_ARRAY_TASK_ID * LINES_PER_JOB ))
END=$(( START + LINES_PER_JOB ))

# ---- RUN THE ASSIGNED LINES ----
i=0
while IFS= read -r line; do
    if (( i >= START && i < END )); then
        # Extract parameters from line
        read r R0 mu localKernel Delta nonLocalProb Nh tmax xmax run <<< "$line"

        echo "Job $SLURM_ARRAY_TASK_ID running parameters: r=$r, R0=$R0, mu=$mu, localKernel=$localKernel, Delta=$Delta, nonLocalProb=$nonLocalProb, Nh=$Nh, tmax=$tmax, xmax=$xmax, run=$run"

        # Run the simulation
        julia ~/coevolution/scripts/1D/phylogenyStudy/cluster/getPhylogenyNonLocalCluster.jl \
              "$r" "$R0" "$mu" "$localKernel" "$Delta" "$nonLocalProb" "$Nh" "$tmax" "$xmax" "$run"
    fi
    ((i++))
done < /home/zayas-orihuela/coevolution/scripts/1D/phylogenyStudy/cluster/params.txt

echo ${SECONDS}