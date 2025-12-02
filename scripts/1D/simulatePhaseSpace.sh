#!/bin/bash
#
# r=(0 1 2 3 4 5 10 15 20 30 40 50 60 80 100)
# R0=(1.05 1.1 1.2 1.3 1.4 1.5 1.8 2 2.4 2.8 3.5 4.2 5 6)

srun julia ~/coevolution/scripts/1D/clusterTrials.jl 40 2 0.2 "Normal(0, 2)" 100000 500