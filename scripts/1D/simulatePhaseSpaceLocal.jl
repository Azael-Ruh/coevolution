include("../../code/simulate/coevolution1DSimulationTools.jl")

rVect = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
R0Vect = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2, 2.4, 2.8, 3.5, 4.2, 5, 6]
mutationRate = 0.2
mutationKernel = Normal(0,2)
Nh::Int64 = 1e6

mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2
vFKKPVect = 2 .* sqrt.((R0Vect .- 1) .* D)

tmax = 500
xmaxVect::Vector{Int64} = max.(500, round.(1.8 .* vFKKPVect .* tmax, digits = -2))

dt = 0.1
dtSampling = 1
initialisation = "steadyState"

plotConfig()

for r in rVect, (idxR0, R0) in enumerate(R0Vect)
    println("Simulating r=$(r), R0 = $(R0)")
    (nx0, hx0, x) = getInitialCondition(initialisation, R0, r, mutationRate, mutationKernel, Nh, xmaxVect[idxR0])
    (nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
    saveSimulation(nx, hx, r, R0, mutationRate, mutationKernel, tmax, dt, xmaxVect[idxR0], initialisation)
    plotSimulationSummary(nx, hx, xmaxVect[idxR0], r, R0)
end