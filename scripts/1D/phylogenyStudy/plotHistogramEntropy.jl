include("src/viralImmuneDistribution.jl")
using LinearAlgebra, JLD2

r = 19
R0 = 1.4

mutationRate = 0.14
nonLocalJump = 35
localKernel = Normal(0, 1)

Nh = 10000000
tmax = 400.
xmax = 200

tAfterNonLocalVector::Vector{Int} = [20, 30, 40, 50, 60, 70, 80, 90, 100]
baseFolder = expanduser("~/PhDVirusImmuneCoEvolution/coevolution")
figDir = abspath(baseFolder * "/figures/genealogicStudies/afterNonLocalEvent")
isdir(figDir) || mkpath(figDir)

cmap = reverse(palette(:viridis, length(tAfterNonLocalVector)))

saveDir = joinpath(baseFolder, "simulations/phylogenyStudy")
nMRCAsamples = 10
NVirus4Times::Int = 2500
nonLocalMutProbVect = [0., 5e-6]
runs = 300

cmapBase = [:gray, :brown3]
entropyVects = Vector{Float64}[]
partialEntropyVects = Vector{Float64}[]
histogramWeightVects = Vector{Vector{Int}}[]
for idxDelta in eachindex(nonLocalMutProbVect)

    maxMRCAtime = 0
    println("Producing plot for Delta = $(nonLocalJump * (nonLocalMutProbVect[idxDelta] != 0)), mu = $(mutationRate)")

    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = dropdims(sum(sampledWeights, dims=1), dims=1)
            maxMRCAtime = max(maxMRCAtime, findlast(totalWeigths .> 0))
        end
    end

    println("Maximum recorded time: $maxMRCAtime")

    entropies = Float64[]
    partialEntropies = Float64[]
    histogramWeights = Vector{Int}[]
    weights = zeros(Integer, maxMRCAtime)
    weigthVar = zeros(Integer, maxMRCAtime)
    for run in 1:runs
        saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mutationRate)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProbVect[idxDelta])tmax$(tmax)nSamples$(nMRCAsamples)NVirus$(NVirus4Times)_$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        if isfile(filePath)
            vars = load(filePath)
            sampledWeights = vars["sampledWeights"]
            totalWeigths = dropdims(sum(sampledWeights, dims=1), dims=1)
            weigthVar += var(sampledWeights, dims=1)[1:length(weights)]
            weights += totalWeigths[1:length(weights)]
            append!(entropies, [entropy(normalize(sampledWeights[i, :], 1)) for i in 1:size(sampledWeights)[1]])
            push!(partialEntropies, entropy(normalize(totalWeigths, 1)))
            append!(histogramWeights, [sampledWeights[i, :] for i in 1:size(sampledWeights)[1]])
        end
    end
    push!(entropyVects, entropies)
    push!(partialEntropyVects, partialEntropies)
    push!(histogramWeightVects, histogramWeights)
    weigthStd = sqrt.(weigthVar)

    finalSensitivity = 10
    finalEdges = push!(collect(0:finalSensitivity:(maxMRCAtime-1)), maxMRCAtime)
    finalHist = fit(Histogram, Float64[], finalEdges)
    finalHist.weights = [sum(weights[(finalEdges[i]+1):finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

    normalisation = sum(weights)
    weigthStd ./= normalisation
    finalWeightStd = [sum(weigthStd[(finalEdges[i]+1):finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

    normalisedHist = normalize(finalHist)

    plotConfig()
    edges = normalisedHist.edges[1]
    weights = normalisedHist.weights
    p = Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], ribbon=[0; finalWeightStd; last(finalWeightStd); 0], xlabel=raw"$T_2$", ylabel=raw"$\mathbb{P}(T_2)$", lw=2, seriestype=:steppost)
    display(p)
end

edges = collect(0:0.05:4)
histogramLocal = fit(Histogram, entropyVects[1], edges)
histogramNonLocal = fit(Histogram, entropyVects[2], edges)
histogramLocal = normalize(histogramLocal)
histogramNonLocal = normalize(histogramNonLocal)

pEntropy = plot([edges[1]; edges; edges[end]], [0; histogramLocal.weights; histogramLocal.weights[end]; 0], lw=2, seriestype=:steppost, c=:steelblue, xlims=(edges[findfirst((histogramLocal.weights .+ histogramNonLocal.weights) .> 0)-1], edges[findlast((histogramLocal.weights .+ histogramNonLocal.weights) .> 0)+1]), label=raw"$\Delta = 0$")
plot!(pEntropy, [edges[1]; edges; edges[end]], [0; histogramNonLocal.weights; histogramNonLocal.weights[end]; 0], lw=2, seriestype=:steppost, c=:coral, label=raw"$\Delta = 35$")
xlabel!(pEntropy, raw"$\mathcal{S}[\mathbb{P}(T_2)]$")
ylabel!(pEntropy, raw"$p(\mathcal{S})$")
vline!(pEntropy, [mean(entropyVects[1])], lw=1, c=:royalblue4, ls=:dash)
vline!(pEntropy, [mean(entropyVects[2])], lw=0.7, c=:darkred, ls=:dash)
display(pEntropy)

histogramLocal = fit(Histogram, partialEntropyVects[1])
histogramNonLocal = fit(Histogram, partialEntropyVects[2])
histogramLocal = normalize(histogramLocal)
histogramNonLocal = normalize(histogramNonLocal)
edgesLocal = histogramLocal.edges[1]
edgesNonLocal = histogramNonLocal.edges[1]

pPartialEntropy = plot([edgesLocal[1]; edgesLocal; edgesLocal[end]], [0; histogramLocal.weights; histogramLocal.weights[end]; 0], lw=2, seriestype=:steppost, c=:steelblue, xlims=(3, 5), label=raw"$\Delta = 0$")
plot!(pPartialEntropy, [edgesNonLocal[1]; edgesNonLocal; edgesNonLocal[end]], [0; histogramNonLocal.weights; histogramNonLocal.weights[end]; 0], lw=2, seriestype=:steppost, c=:coral, label=raw"$\Delta = 35$")
xlabel!(pPartialEntropy, raw"$\mathcal{S}[\mathbb{P}(T_2\,| T_2<T_r)]$")
ylabel!(pPartialEntropy, raw"$p(\mathcal{S})$")
# vline!(pPartialEntropy, [mean(partialEntropyVects[1])], lw = 1, c = :royalblue4, ls = :dash)
# vline!(pPartialEntropy, [mean(partialEntropyVects[2])], lw = 0.7, c = :darkred, ls = :dash)