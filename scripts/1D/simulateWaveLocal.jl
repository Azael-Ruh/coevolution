include("../../code/simulate/coevolution1DSimulationTools.jl")

r = 5
R0 = 1.5
mutationRate = 0.2
xmax::Int64 = 2000
Nh::Int64 = 1e6

nonLocalJump = 40
nonLocalMutProb = 1e-6
localKernel = Normal(0,2)
mutationKernel = Normal(0,2) # piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

tmax = 800
dt = 0.1
dtSampling = 1

(Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxLoc, hxLoc) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

(nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

p = plotSimulationSummary(nx, hx, xmax, r, R0)