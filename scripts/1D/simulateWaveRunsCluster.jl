include("../../code/simulate/coevolution1DSimulationTools.jl")

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
mutationRate = parse(Float64, ARGS[3])
mutationKernel = eval(Meta.parse(ARGS[4])) # Dangerous
Nh = parse(Int, ARGS[5])

mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2
vFKKP = 2 * sqrt.((R0 - 1) * D)

tmax = parse(Int, ARGS[6])
xmax = max.(500, round.(1.8 * vFKKP * tmax, digits = -2))

dt = 0.1
dtSampling = 1
initialisation = "steadyState"

nRuns = parse(Int, ARGS[7])

println("Simulating r=$(r), R0 = $(R0) for $(nRuns) runs")
for i in 1:nRuns
    (nx0, hx0, x) = getInitialCondition(initialisation, R0, r, mutationRate, mutationKernel, Nh, xmax)
    (nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
    saveSimulation(nx, hx, r, R0, mutationRate, mutationKernel, tmax, dt, xmax, initialisation, baseFolder = "/home/zayas-orihuela/coevolution/", fileAppend = "_run$(i)")
end

