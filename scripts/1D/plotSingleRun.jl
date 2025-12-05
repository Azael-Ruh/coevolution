include("../../code/simulate/coevolution1DSimulationTools.jl")

rVect = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
R0Vect = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2, 2.4, 2.8, 3.5, 4.2, 5, 6, 7, 8, 9, 10]
r = rVect[end]
R0 = R0Vect[end]
mutationRate = 0.2
mutationKernel = Normal(0, 2)
tmax = 1000
run = 10
Nh = 100000

dt = 0.1
dtSampling = 1
initialisation = "steadyState"

nx, hx, x = loadSimulationDistributionData(R0, r, Nh, mutationRate, mutationKernel, tmax, dt, dtSampling, initialisation; baseFolder = "/home/zayas-orihuela/coevolution/", fileAppend = "_run$(run)")
xmax = x[end]
p = plotSimulationSummary(nx, hx, xmax, r, R0; tTransient = 100, dtSampling = 1)

baseFolder = "/home/zayas-orihuela/coevolution/"
dist = kernType(mutationKernel) * "$(std(mutationKernel))"
figDir = baseFolder * "figures/1D/" * dist * "/dt$(dt)_dtSamp$(dtSampling)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_" * initialisation

isdir(figDir) || mkpath(figDir)
savefig(p, joinpath(figDir, "simulationSummary_run$(run).png"))