include("../../code/simulate/coevolution1DSimulationTools.jl")

r = parse(Float64, ARGS[1])
R0 = parse(Float64, ARGS[2])
mutationRate = parse(Float64, ARGS[3])
mutationKernel = eval(Meta.parse(ARGS[4])) # Dangerous
Nh = parse(Int, ARGS[5])

mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2
vFKKPVect = 2 * sqrt.((R0 - 1) * D)

tmax = parse(Float64, ARGS[6])
xmaxVect::Vector{Int64} = max.(500, round.(1.8 .* vFKKPVect .* tmax, digits = -2))

dt = 0.1
dtSampling = 1
initialisation = "steadyState"

plotConfig()

println("Simulating r=$(r), R0 = $(R0)")
(nx0, hx0, x) = getInitialCondition(initialisation, R0, r, mutationRate, mutationKernel, Nh, xmaxVect[idxR0])
(nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
saveSimulation(nx, hx, r, R0, mutationRate, mutationKernel, tmax, dt, xmaxVect[idxR0], initialisation, baseFolder = "~/coevolution/")

