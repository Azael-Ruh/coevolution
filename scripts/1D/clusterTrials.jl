include("../../code/simulate/coevolution1DSimulationTools.jl")
using Distributions

r = parse(Float64, ARGS[1])
R0 = parse(Float64, ARGS[2])
mutationRate = parse(Float64, ARGS[3])
mutationKernel = eval(Meta.parse(ARGS[4])) # Dangerous
# mutationKernel = getfield(Distributions, Symbol(mutationKernelType))(mutationKernelParams...)
Nh = parse(Int, ARGS[5])
tmax = parse(Float64, ARGS[6])

println("Found r = $r of type $(typeof(r)) R0 = $R0 mu = $mutationRate muK = $mutationKernel (of type $(typeof(mutationKernel))) Nh = $Nh tmax = $tmax")



