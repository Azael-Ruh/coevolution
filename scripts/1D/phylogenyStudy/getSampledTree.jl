include("./src/viralImmuneDistribution.jl")
using LinearAlgebra

function forgetInitalCondition!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Int

    nOriginalNodes = sum(getBirthTime.(vcat(viDist.viralPop.viralNodes...)) .== 0)
    println("$nOriginalNodes original nodes!")
    run = 1

    while nOriginalNodes > 0
        println("Run $run before forgetting initial condition")
        time = ((run-1)*simSet.tmax):simSet.dt:(run*simSetDist.tmax-simSet.dt)
        survivalFlag = true
        for t in time
            survivalFlag, _ = simulationStep!(viDist, mParams, simSet, t)
            survivalFlag || (println("WARNING: virus extinct"); break)
        end

        survivalFlag ? translateDistributionBackLeft!(viDist, mParams) : (global viDist=viralImmuneDistribution(x, nx0, hx0); getGrowthRate!(viDist, mParams); run -= 1)

        nOriginalNodes = sum(getBirthTime.(vcat(viDist.viralPop.viralNodes...)) .== 0)
        println("$nOriginalNodes original nodes!")
        run += 1
    end

    return run
end

function runSamplingSimulation!(viDist::viralImmuneDistribution, mParams::modelParams, simSetDist::simulationConfig, simSetNode::simulationConfig, sample::Int)::Bool

    time = ((sample-1)*simSetDist.tmax):simSetDist.dt:(sample*simSetDist.tmax-simSetDist.dt)
    survivalFlag = true

    for t in time
        survivalFlag, nonLoalFlag = simulationStep!(viDist, mParams, simSetDist, t)
        survivalFlag || (println("WARNING: virus extinct"); break)
        nonLoalFlag && println("Non local event!")

        isTime2Sample(t, simSetNode) && sampleNvirus!(viDist, Nsampling)
        isTime2Sample(t, simSetDist) && sampleDistributions!(viDist, sampDists)
    end

    return survivalFlag
end

r = 50
R0 = 1.45
Nh::Int = 3e7
mu = 0.2
nonLocalMutProb = 0
nonLocalJump = 35
localKernel = Normal(0, 1)
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 400
dt = 0.1
dtNodeSampling = 0.5
dtDistSampling = 4
Nsampling = 1
simSetNode = simulationConfig(tmax, dt, dtNodeSampling)
simSetDist = simulationConfig(tmax, dt, dtDistSampling)
sampDists = sampledResults()

xmax = 250
(nx0::Vector{Int}, hx0::Vector{Int}, x) = getInitialCondition("steadyState", R0, r, mu, mutationKernel, Nh, xmax)
x = first(x):last(x)

viDist = viralImmuneDistribution(x, nx0, hx0)
getGrowthRate!(viDist, mParams)

run = forgetInitalCondition!(viDist, mParams, simSetDist)

sampleDistributions!(viDist, sampDists)
nSamples = 3

for sample in run:(run+nSamples-1)
    println("Starting simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    survivalFlag = runSamplingSimulation!(viDist::viralImmuneDistribution, mParams::modelParams, simSetDist::simulationConfig, simSetNode::simulationConfig, sample::Int)

    println("Finished simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    survivalFlag ? translateDistributionBackLeft!(viDist, mParams) : (global viDist=viralImmuneDistribution(x, nx0, hx0); getGrowthRate!(viDist, mParams))

    println("Translated distrbution back to xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
end

plotConfig()
sampledNodes = getAllSampledNodes(viDist.viralPop);
sc = Plots.scatter(getBirthTime.(sampledNodes), getDivergence.(sampledNodes), xlabel=raw"Sampling time", ylabel=raw"Divergence from root", ms=2, msc=:steelblue, mc=:steelblue)
Plots.plot!(twinx(), ((run-1)*simSetDist.tmax):simSetDist.dtSampling:((run+nSamples-1)*simSetDist.tmax), sum.(sampDists.nxt), c=:coral, ylabel=raw"Total infected population", yscale=:log10)
display(sc)

fullSampledTree = getFullSampledTreeCopy(viDist.viralPop::viralPopulation);
ladderize!(fullSampledTree)
cfull = [normalisedCollessLikeLogMDM(node) for node in PreOrderDFS(fullSampledTree)]
fig, ax, tp = plotTreeFromNode(fullSampledTree; branchcolor=cfull)
Colorbar(fig[1, 2], limits=extrema(cfull), colormap=:viridis, label="Normalised Colless-like index")
pts = [tp.nodepoints[][n] for n in PreOrderDFS(fullSampledTree) if BasicTreePlots.isleaf(n)]
cleaves = [node.divergenceFromRoot for node in PreOrderDFS(fullSampledTree) if BasicTreePlots.isleaf(node)]
CairoMakie.scatter!(pts, markersize=4, color=cleaves, colorrange=extrema(cleaves), colormap=:inferno)
Colorbar(fig[1, 3], limits=extrema(cleaves), colormap=:inferno, label="Divergence from root")
display(fig)

MRCAtimes = getMRCAtimes(sampledNodes)
normalisedHist = normalize(fit(Histogram, MRCAtimes))
edges = normalisedHist.edges[1]
weights = normalisedHist.weights
Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel=raw"$T_2$", ylabel=raw"$\mathbb{P}(T_2)$", lw=2, seriestype=:steppost)