include("../../code/simulate/coevolution1DSimulationTools.jl")

r = 40
R0 = 1.2
mutationRate = 0.3
xmax::Int64 = 1000
Nh::Int64 = 1e7

nonLocalJump = 0
nonLocalMutProb = 1e-6
localKernel = Normal(0,1)
mutationKernel = Normal(0,1) # piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

tmax = 2000
dt = 0.1
dtSampling = 2

(Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxLoc, hxLoc) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

(nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

p = plotSimulationSummary(nx, hx, xmax, r, R0)
# g = animateSimulation(nx, hx, x, Nh)