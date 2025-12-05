include("../../code/simulate/coevolution1DSimulationTools.jl")
using JLD2

rVect = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
R0Vect = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2, 2.4, 2.8, 3.5, 4.2, 5, 6, 7, 8, 9, 10]
mutationRate = 0.2
mutationKernel = Normal(0, 2)
tmax = 1000
nRuns = 10
Nh = 100000

dt = 0.1
dtSampling = 1
initialisation = "steadyState"

survivalProb, vAverage, vStd, NAverage, NStd = producePhaseMatrixes(R0Vect, rVect, nRuns, Nh, mutationRate, mutationKernel, tmax, dt, dtSampling, initialisation, bFolder = "/home/zayas-orihuela/coevolution/")
save("/home/zayas-orihuela/coevolution/simulations/1D/Normal2.0/phaseSpaceMatrixes.jld2", "sProb", survivalProb, "vAvg", vAverage, "vStd", vStd, "NAvg", NAverage, "NStd", NStd)
p = plotPhaseDiagrams(R0Vect, rVect, survivalProb, vAverage, vStd, NAverage, NStd, mutationKernel, baseFolder = "/home/zayas-orihuela/coevolution/")