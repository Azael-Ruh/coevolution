include("../src/viralImmuneDistribution.jl")
using LinearAlgebra, JLD2

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
Nh::Int = parse(Float64, ARGS[7])
mu = parse(Float64, ARGS[3])
localKernel = eval(Meta.parse(ARGS[4]))
nonLocalJump = parse(Int, ARGS[5])
nonLocalMutProb = parse(Float64, ARGS[6])
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = parse(Float64, ARGS[8])
dt = 0.1
simSet = simulationConfig(tmax, dt)

xmax::Int = parse(Int, ARGS[9])
(nx0::Vector{Int}, hx0::Vector{Int}, x) = getInitialCondition("steadyState", R0, r, mu, mutationKernel, Nh, xmax)
x = first(x):last(x)

viDist = viralImmuneDistribution(x, nx0, hx0)
getGrowthRate!(viDist, mParams)

nMRCAsamples = 10
NVirus4Times = 2500
histogramEdges = 0:1:tmax*nMRCAsamples
MRCAtimesHistogram = fit(Histogram, Float64[], histogramEdges)
sampledWeights = Matrix{Int64}(undef, nMRCAsamples, length(histogramEdges) - 1)

#TODO: check for extinction! -> Check it works! -> Does not seem to be working!!!
for sample in 1:nMRCAsamples
    println("Starting sample $sample at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    time = ((sample - 1)*simSet.tmax ):simSet.dt:(sample*simSet.tmax - simSet.dt)
    
    simulationFailed = false
    @time for t in time
        simulationStep!(viDist, mParams, simSet, t) || (simulationFailed = true; break)
    end # ~

    while simulationFailed
        println("WARNING: virus extinct. Restarting simulation")
        global viDist = viralImmuneDistribution(x, nx0, hx0)
        getGrowthRate!(viDist, mParams)
        
        println("Starting sample $sample at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
        simulationFailed = false
        @time for t in time
            simulationStep!(viDist, mParams, simSet, t) || (simulationFailed = true; break)
        end
    end

    println("Finished simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    translateDistributionBackLeft!(viDist, mParams)

    println("Translated distrbution back to xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    @time newMRCAtimes = getMRCAtimes(viDist.viralPop, (NVirus4Times > sum(viDist.nx) ? sum(viDist.nx) : NVirus4Times))
    any(newMRCAtimes == Inf) && println("Initial condition still not forgotten")

    newHist = fit(Histogram, newMRCAtimes, histogramEdges)
    sampledWeights[sample, :] = newHist.weights
end

run = parse(Int, ARGS[10])
saveDir = expanduser("~/coevolution/simulations/phylogenyStudy")
saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mu)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProb)tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
jldsave(joinpath(saveDir, saveFile); sampledWeights)