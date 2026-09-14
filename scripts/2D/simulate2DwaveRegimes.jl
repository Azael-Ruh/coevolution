include("./src/viralImmune2D.jl")

using JLD2

r = 30
R0 = 2.5
Nh::Int = 1e7
muVect = [0.10, 0.12, 0.14]
nonLocalMutProb = 2e-6
nonLocalJumpVector = [0, 10, 20, 30, 40, 50, 60]
localMutationKernel = NextNeighbour()

for mu in muVect, nonLocalJump in nonLocalJumpVector
    mutationKernel = piecewiseKernel2D(nonLocalMutProb, nonLocalJump, localMutationKernel)
    mParams = modelParams(r, R0, Nh, mu, mutationKernel)

    tmax = 250
    dt = 0.1
    dtSampling = 2
    simSet = simulationConfig(tmax, dt, dtSampling)
    sampDists = sampledResults()

    x = -150:150
    y = -150:150

    vi2D = viralImmuneDistribution2D(x, y)
    getInitialDistribution!(vi2D, mParams, sampDists)

    nRuns = 10
    speciationTimes = Float64[]
    extinctionTimes = Float64[]
    nonLocalTimes = Float64[]

    println("Starting simulation with mu = $mu, Delta = $nonLocalJump")

    for run in 1:nRuns 
        println("Run $run of $nRuns")

        time = dt:dt:tmax
        extinctionFlag = false

        for t in time
            extinctionFlag, nonLocalFlag = simulationStep!(vi2D, mParams, simSet) 
            
            extinctionFlag && (println("WARNING: virus extinct"); push!(extinctionTimes, t); break)
            nonLocalFlag && (push!(nonLocalTimes, t); println("Non-local event!"))
            
            if isTime2Sample(t, simSet)
                println("t = $t")
                sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
                hasSpeciationHappened(sampDists) && push!(speciationTimes, t)
                hasExtinctionHappened(sampDists) && push!(extinctionTimes, t)
            end
        end

        extinctionFlag ?
            getInitialDistribution!(vi2D, mParams, sampDists) :
            isolateBestSpeciesBack2Origin!(vi2D, mParams)
    end

    nSpeciationEvents = length(speciationTimes) #TODO: improve for speciation detection!
    nExtinctionEvents = length(extinctionTimes)
    nNonLocalEvents = length(nonLocalTimes)

    saveDir = expanduser("~/PhDVirusImmuneCoEvolution/coevolution/simulations/2D/speciationNExtinction")
    saveFile = "speciationExtinction_r$(r)R0$(R0)mu$(mu)tmax$(tmax)totalRuns$(nRuns)nonLocalProb$(nonLocalMutProb)Delta$(mutationKernel.nonLocalJump).jld2"
    jldsave(joinpath(saveDir, saveFile); nSpeciationEvents, speciationTimes, nExtinctionEvents, extinctionTimes, nNonLocalEvents, nonLocalTimes)

    # plotAnimation(sampDists, vi2D.x, vi2D.y)
end