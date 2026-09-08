include("./src/viralImmuneDistribution.jl")
using LinearAlgebra

r = 19
R0 = 1.4
Nh::Int = 1e7
mu = 0.14
localKernel = Normal(0,1)
nonLocalJump = 35
nonLocalMutProb = 5e-6
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 500
dt = 0.1
simSet = simulationConfig(tmax, dt)

xmax = 250
nx0::Vector{Int}, hx0::Vector{Int}, x = getInitialCondition("steadyState", R0, r, mu, mutationKernel, Nh, xmax)
x = first(x):last(x)

viDist = viralImmuneDistribution(x, nx0, hx0)
getGrowthRate!(viDist, mParams)

nMaxRuns = 10
nMinSamples = 10
NVirus4Times = 2000
histogramEdges = 0:1:tmax*nMaxRuns
MRCAtimesHistogram = fit(Histogram, Float64[], histogramEdges)
sampledWeights = Vector{Int64}[]

time2sampleAfterNonLocalEvent = 85
nonLocalDelayVector = Float64[]

nSamplesDone = 0
for run in 1:nMaxRuns
    println("Starting simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    time = ((run - 1)*simSet.tmax ):simSet.dt:(run*simSet.tmax - simSet.dt)

    simulationFailed = false
    @time for t in time
        survivalFlag, nonLocalFlag = simulationStep!(viDist, mParams, simSet, t) 
        
        survivalFlag || (simulationFailed = true; break)

        # Update nonLocalDelayVector and sample if the first element is zero
        if !isempty(nonLocalDelayVector)
            nonLocalDelayVector .-= dt
            
            if first(nonLocalDelayVector) <= 0
                @time newMRCAtimes = getMRCAtimes(viDist.viralPop, min(NVirus4Times, sum(viDist.nx)))
                filter!(x -> x < Inf, newMRCAtimes)
                newHist = fit(Histogram, newMRCAtimes, histogramEdges)
                push!(sampledWeights, newHist.weights)

                global nSamplesDone += 1

                popfirst!(nonLocalDelayVector)
            end
        end

        # If there is a non-local event, sample after `time2sampleAfterNonLocalEvent`
        nonLocalFlag && (push!(nonLocalDelayVector, time2sampleAfterNonLocalEvent); println("Non local event!"))

    end

    while simulationFailed
        println("WARNING: virus extinct. Restarting simulation")
        global viDist = viralImmuneDistribution(x, nx0, hx0)
        getGrowthRate!(viDist, mParams)
        
        global nonLocalDelayVector = Float64[]
        
        println("Starting sample $sample at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
        simulationFailed = false

        @time for t in time
            survivalFlag, nonLocalFlag = simulationStep!(viDist, mParams, simSet, t) 
    
            survivalFlag || (simulationFailed = true; break)

            # Update nonLocalDelayVector and sample if the first element is zero
            if !isempty(nonLocalDelayVector)
                nonLocalDelayVector .-= dt
                
                if first(nonLocalDelayVector) <= 0
                    @time newMRCAtimes = getMRCAtimes(viDist.viralPop, min(NVirus4Times, sum(viDist.nx)))
                    filter!(x -> x < Inf, newMRCAtimes)
                    newHist = fit(Histogram, newMRCAtimes, histogramEdges)
                    push!(sampledWeights, newHist.weights)

                    global nSamplesDone += 1

                    popfirst!(nonLocalDelayVector)
                end
            end

        # If there is a non-local event, sample after `time2sampleAfterNonLocalEvent`
        nonLocalFlag && (push!(nonLocalDelayVector, time2sampleAfterNonLocalEvent); println("Non local event!"))
        end
         
    end
    println("Finished simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    if nSamplesDone > nMinSamples
        println("Minimum desired number of samples achieved. Final result: &(nSamplesDone) samples > $(nMinSamples) minimum desired samples.")
        break
    end

    translateDistributionBackLeft!(viDist, mParams)

    println("Translated distrbution back to xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
end

weights = sum(sampledWeights)
maxMRCAtime = findlast(weights .> 0)
finalSensitivity = 10
finalEdges = push!(collect(0:finalSensitivity:maxMRCAtime - 1), maxMRCAtime)
finalHist = fit(Histogram, Float64[], finalEdges)
finalHist.weights = [sum(weights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]

normalisedHist = normalize(finalHist)

plotConfig()
edges = normalisedHist.edges[1]
weights = normalisedHist.weights
Plots.plot([edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)

p = Plots.plot([],[], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", seriestype = :steppost)
for samp in eachindex(sampledWeights)
    h = fit(Histogram, Float64[], finalEdges)
    originalWeights = sampledWeights[samp]
    newWeights = [sum(originalWeights[finalEdges[i] + 1:finalEdges[i+1]]) for i in 1:(length(finalEdges)-1)]
    h.weights = newWeights
    normalisedH = normalize(h)
    weights = normalisedH.weights
    Plots.plot!(p, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], c = :gray, lw = 0.5, alpha = 0.4, seriestype = :steppost)
end
weights = normalisedHist.weights
Plots.plot!(p, [edges[1]; edges; edges[end]], [0; weights; last(weights); 0], xlabel = raw"$T_2$", ylabel = raw"$\mathbb{P}(T_2)$", lw = 2, seriestype = :steppost)