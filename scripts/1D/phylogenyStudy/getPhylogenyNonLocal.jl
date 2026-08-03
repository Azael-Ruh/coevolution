include("./src/viralImmuneDistribution.jl")
using LinearAlgebra

r = 18
R0 = 1.35
Nh::Int = 1e7
mu = 0.17
localKernel = Normal(0,1)
nonLocalJump = 30
nonLocalMutProb = 5e-6
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 400
dt = 0.1
simSet = simulationConfig(tmax, dt)

xmax::Int = 250
(nx0::Vector{Int}, hx0::Vector{Int}, x) = getInitialCondition("steadyState", R0, r, mu, mutationKernel, Nh, xmax)
x = first(x):last(x)

viDist = viralImmuneDistribution(x, nx0, hx0)
getGrowthRate!(viDist, mParams)

nMRCAsamples = 20
histogramEdges = 0:1:tmax*nMRCAsamples
MRCAtimesHistogram = fit(Histogram, Float64[], histogramEdges)
sampledWeights = Matrix{Int64}(undef, nMRCAsamples, length(histogramEdges) - 1)

for sample in 1:nMRCAsamples
    println("Starting sample $sample at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    time = ((sample - 1)*simSet.tmax ):simSet.dt:(sample*simSet.tmax - simSet.dt)
    @time for t in time
        simulationStep!(viDist, mParams, simSet, t) || (println("WARNING: virus extinct"); break)
    end # ~

    println("Finished simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    translateDistributionBackLeft!(viDist, mParams)

    println("Translated distrbution back to xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    @time newMRCAtimes = getMRCAtimes(viDist.viralPop, 2000)
    any(newMRCAtimes == Inf) && println("Initial condition still not forgotten")

    newHist = fit(Histogram, newMRCAtimes, histogramEdges)
    sampledWeights[sample, :] = newHist.weights
end

weights = dropdims(sum(sampledWeights, dims = 1), dims = 1)
maxMRCAtime = findlast(weights .> 0)+1
finalSensitivity = 18
finalEdges = push!(collect(0:finalSensitivity:maxMRCAtime), maxMRCAtime)
finalHist = fit(Histogram, Float64[], finalEdges)
finalHist.weights = [sum(weights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

normalisedHist = normalize(finalHist)

plotConfig()
edges = normalisedHist.edges[1]
weights = normalisedHist.weights
p0 = Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)

p = Plots.plot([],[], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", seriestype = :steppost)
for samp in 1:nMRCAsamples
    h = fit(Histogram, Float64[], finalEdges)
    originalWeights = sampledWeights[samp, :]
    newWeights = [sum(originalWeights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]
    h.weights = newWeights
    normalisedH = normalize(h)
    weights = normalisedH.weights
    Plots.plot!(p, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], c = :gray, lw = 0.5, alpha = 0.4, seriestype = :steppost)
end
weights = normalisedHist.weights
Plots.plot!(p, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)