include("code/simulate/coevolution1DSimulationTools.jl")

r = 0
R0 = 1.05
mutationRate = 0.2
mutationKernel = Normal(0,2)
xmax::Int64 = 500
Nh::Int64 = 1e6

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

tmax = 500
dt = 0.1
dtSampling = 1

(nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

p = plotSimulationSummary(nx, hx, xmax, r, R0)