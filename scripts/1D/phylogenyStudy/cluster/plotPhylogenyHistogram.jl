include("../src/viralImmuneDistribution.jl")
using LinearAlgebra, JLD2

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])

mutationRateVect = let expr = Meta.parse(ARGS[3])
    @assert expr.head == :vect
    Float64.(expr.args)
end
nonLocalMutProbVect = let expr = Meta.parse(ARGS[4])
    @assert expr.head == :vect
    Float64.(expr.args)
end
nonLocalJumpVect = let expr = Meta.parse(ARGS[5])
    @assert expr.head == :vect
    Int.(expr.args)
end
localKernel = eval(Meta.parse(ARGS[6])) # Dangerous

Nh = parse(Int, ARGS[7])
tmax = parse(Float64, ARGS[8])
xmax = parse(Int, ARGS[9])
runs = parse(Int, ARGS[10])

saveDir = expanduser("~/coevolution/simulations/phylogenyStudy")
nMRCAsamples = 5
NVirus4Times = 2000

baseFolder = expanduser("~/coevolution/")
figDir = baseFolder * "figures/genealogicStudies/"
isdir(figDir) || mkpath(figDir)
pTot = Plots.plot()

for mu in mutationRateVect, idxDelta in eachindex(nonLocalJumpVect)

    maxMRCAtime = 0
    println("Producing plot for Delta = $(nonLocalJumpVect[idxDelta]), mu = $(mu)")
    
    for run in [collect(1:2); collect(4:runs)]
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mu)Delta$(nonLocalJumpVect[idxDelta])nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = dropdims(sum(sampledWeights, dims = 1), dims = 1)
            maxMRCAtime = max(maxMRCAtime, findlast(totalWeigths .> 0))
        end
    end

    println("Maximum recorded time: $maxMRCAtime")

    weights = zeros(Integer, maxMRCAtime)
    for run in [collect(1:2); collect(4:runs)]
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mu)Delta$(nonLocalJumpVect[idxDelta])nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = dropdims(sum(sampledWeights, dims = 1), dims = 1)
            weights += totalWeigths[1:length(weights)]
        end
    end

    finalSensitivity = 5
    finalEdges = push!(collect(0:finalSensitivity:maxMRCAtime-1), maxMRCAtime)
    finalHist = fit(Histogram, Float64[], finalEdges)
    finalHist.weights = [sum(weights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

    normalisedHist = normalize(finalHist)

    plotConfig()
    edges = normalisedHist.edges[1]
    weights = normalisedHist.weights
    p0 = Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)

    savefig(p0, joinpath(figDir, "T2histogram_r$(r)R0$(R0)Delta$(nonLocalJumpVect[idxDelta])mu$(mu)nSamples$(nMRCAsamples*runs)NVirus$(NVirus4Times).png"))

    Plots.plot!(pTot, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost, label = raw"$\Delta = " * "$(nonLocalJumpVect[idxDelta])" * raw"$")
end

savefig(pTot, joinpath(figDir, "T2histogram_r$(r)R0$(R0)nSamples$(nMRCAsamples*runs)NVirus$(NVirus4Times).png"))