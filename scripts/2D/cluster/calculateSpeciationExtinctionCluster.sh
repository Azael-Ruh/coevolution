#!/bin/bash
#SBATCH -p R-Infinite
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err

#SBATCH --array=0-95        # 96 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -t 0-48:00

# ---- CONFIG ----
SECONDS=0
MUSIZE=6
DELTASIZE=8
RUNS=2
TOTAL_LINES=$(( MUSIZE * DELTASIZE * RUNS))
NJOBS=96
LINES_PER_JOB=$(( TOTAL_LINES / NJOBS))
START=$(( SLURM_ARRAY_TASK_ID * LINES_PER_JOB ))
END=$(( START + LINES_PER_JOB ))

# ---- JULIA ENV INFO ----
echo "HOST: $(hostname)"
echo "JULIA: $(which julia)"
julia --version
echo "JULIA_PROJECT=$JULIA_PROJECT"

~/.juliaup/bin/julia -e '
using Pkg
println("active project: ", Base.active_project())
println("DEPOT_PATH: ", DEPOT_PATH)
println("LOAD_PATH: ", LOAD_PATH)
Pkg.status()
'

# ---- RUN THE ASSIGNED LINES ----
i=0
while IFS= read -r line; do
    if (( i >= START && i < END )); then
        # Extract parameters from line
        read r R0 mu nonLocalJump nonLocalMutProb Nh tmax nCycles run <<< "$line"

        echo "Job $SLURM_ARRAY_TASK_ID running parameters: r=$r, R0=$R0, mu=$mu, Delta=$nonLocalJump, nonLocalMutProb=$nonLocalMutProb Nh=$Nh, tmax=$tmax, nCycles=$nCycles, run=$run"

        # Run the simulation
        ~/.juliaup/bin/julia ~/coevolution/scripts/2D/cluster/calculateSpeciationExtinctionCluster.jl \
              "$r" "$R0" "$Nh" "$mu" "$nonLocalMutProb" "$nonLocalJump" "$tmax" "$nCycles" "$run"
    fi
    ((i++))
done < /home/zayas-orihuela/coevolution/scripts/2D/cluster/params.txt

echo ${SECONDS}