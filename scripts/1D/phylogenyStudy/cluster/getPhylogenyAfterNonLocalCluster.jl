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

        while simulationFailed
            println("WARNING: virus extinct. Restarting simulation")
            global viDist = viralImmuneDistribution(x, nx0, hx0)
            getGrowthRate!(viDist, mParams)
            
            println("Starting sample $sample at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
            simulationFailed = false
            @time for t in time
                extinctionFlag, nonLocalFlag = simulationStep!(viDist, mParams, simSet, t) 
        
                extinctionFlag || (simulationFailed = true; break)

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
            end

        # If there is a non-local event, sample after `time2sampleAfterNonLocalEvent`
        nonLocalFlag && (push!(nonLocalDelayVector, time2sampleAfterNonLocalEvent); println("Non local event!"))
    end

    println("Finished simulation at xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")

    if nSamplesDone > nMinSamples
        println("Minimum desired number of samples achieved. Final result: &(nSamplesDone) samples > $(nMinSamples) minimum desired samples.")
        break
    end

    translateDistributionBackLeft!(viDist, mParams)

    println("Translated distrbution back to xAv = $(sum(x .* viDist.nx) ./ sum(viDist.nx))")
end

run = parse(Int, ARGS[10])
saveDir = expanduser("~/coevolution/simulations/phylogenyStudy/afterNonLocalEvent")
saveFile = "sampledMRCAtimeWeights_r$(r)R0$(R0)mu$(mu)Delta$(nonLocalJump)nonLocalProb$(nonLocalMutProb)tmax$(tmax)tAfterNonLocal$(time2sampleAfterNonLocalEvent)nSamples$(nMinSamples)NVirus$(NVirus4Times)_$(run).jld2"
jldsave(joinpath(saveDir, saveFile); sampledWeights)