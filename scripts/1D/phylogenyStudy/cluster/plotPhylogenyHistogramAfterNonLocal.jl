include("../src/viralImmuneDistribution.jl")
using LinearAlgebra, JLD2

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])

mutationRate = parse(Float64, ARGS[3])
nonLocalMutProbVect = let expr = Meta.parse(ARGS[4])
    @assert expr.head == :vect
    Float64.(expr.args)
end
nonLocalJump = parse(Int, ARGS[5])
localKernel = eval(Meta.parse(ARGS[6])) # Dangerous

Nh = parse(Int, ARGS[7])
tmax = parse(Float64, ARGS[8])
xmax = parse(Int, ARGS[9])
runs = parse(Int, ARGS[10])

saveDir = expanduser("~/coevolution/simulations/phylogenyStudy/afterNonLocalEvent")
tAfterNonLocalVector::Vector{Int} = [60]
nMinSamples::Int = 10
NVirus4Times::Int = 2000

baseFolder = expanduser("~/coevolution")
figDir = baseFolder * "/figures/genealogicStudies/afterNonLocalEvent"
isdir(figDir) || mkpath(figDir)
pTot = Plots.plot()

for idxDelta in eachindex(nonLocalMutProbVect), tAfterNonLocal in tAfterNonLocalVector 

    maxMRCAtime = 0
    println("Producing plot for Delta = $(nonLocalJump * (nonLocalMutProbVect[idxDelta] != 0)), mu = $(mutationRate)")
    
    nFiles = 0
    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)tAfterNonLocal$(tAfterNonLocal)nSamples$(nMinSamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            nFiles += 1
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = sum(sampledWeights)
            maxMRCAtime = max(maxMRCAtime, findlast(totalWeigths .> 0))
        end
    end

    println("Found $nFiles files. Maximum recorded time: $maxMRCAtime")

    weights = zeros(Integer, maxMRCAtime)
    weigthVar = zeros(Integer, maxMRCAtime)
    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)tAfterNonLocal$(tAfterNonLocal)nSamples$(nMinSamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = sum(sampledWeights)
            weigthVar += var(stack(sampledWeights, dims = 1), dims = 1)[1:length(weights)]
            weights += totalWeigths[1:length(weights)]
        end
    end
    weigthStd = sqrt.(weigthVar)

    finalSensitivity = 7
    finalEdges = push!(collect(0:finalSensitivity:maxMRCAtime-1), maxMRCAtime)
    finalHist = fit(Histogram, Float64[], finalEdges)
    finalHist.weights = [sum(weights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]
    
    normalisation = sum(weights)
    weigthStd ./= normalisation
    finalWeightStd = [sum(weigthStd[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

    normalisedHist = normalize(finalHist)

    plotConfig()
    edges = normalisedHist.edges[1]
    weights = normalisedHist.weights
    p0 = Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], ribbon = [0; finalWeightStd; last(finalWeightStd); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)

    savefig(p0, joinpath(figDir, "T2histogramAfterNonLocal_r$(r)R0$(R0)Delta$(nonLocalJump * (nonLocalMutProbVect[idxDelta] != 0))mu$(mutationRate)tAfterNonLocal$(tAfterNonLocal)nMinSamples$(nMinSamples*runs)NVirus$(NVirus4Times).svg"))

    Plots.plot!(pTot, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], ribbon = [0; finalWeightStd; last(finalWeightStd); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost, label = raw"$\Delta = 35, T_\mathrm{NL}=40$", xlims = (0, min(1000, maxMRCAtime)))
    Plots.vline!([tAfterNonLocal], lw = 1, ls = :dash, c = :gray, label = raw"$T_\mathrm{NL}=" * "$tAfterNonLocal" * raw"$")
end

saveDir = expanduser("~/coevolution/simulations/phylogenyStudy")
nMRCAsamples = 10
NVirus4Times = 2500
nonLocalMutProbVect = [0., 5e-6]
runs = 200
for idxDelta in eachindex(nonLocalMutProbVect)

    maxMRCAtime = 0
    println("Producing plot for Delta = $(nonLocalJump * (nonLocalMutProbVect[idxDelta] != 0)), mu = $(mutationRate)")
    
    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
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
    weigthVar = zeros(Integer, maxMRCAtime)
    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = dropdims(sum(sampledWeights, dims = 1), dims = 1)
            weigthVar += var(sampledWeights, dims = 1)[1:length(weights)]
            weights += totalWeigths[1:length(weights)]
        end
    end
    weigthStd = sqrt.(weigthVar)

    finalSensitivity = 7
    finalEdges = push!(collect(0:finalSensitivity:maxMRCAtime-1), maxMRCAtime)
    finalHist = fit(Histogram, Float64[], finalEdges)
    finalHist.weights = [sum(weights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]
    
    normalisation = sum(weights)
    weigthStd ./= normalisation
    finalWeightStd = [sum(weigthStd[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

    normalisedHist = normalize(finalHist)

    plotConfig()
    edges = normalisedHist.edges[1]
    weights = normalisedHist.weights
    Plots.plot!(pTot, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], ribbon = [0; finalWeightStd; last(finalWeightStd); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost, label = raw"$\Delta = 0$", xlims = (0, min(1000, maxMRCAtime)))
end

savefig(pTot, joinpath(figDir, "T2histogramAfterNonLocal_r$(r)R0$(R0)Delta$(nonLocalJump)mu$(mutationRate)nMinSamples$(nMinSamples*runs)NVirus$(NVirus4Times).svg"))