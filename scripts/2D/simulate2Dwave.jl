include("./src/viralImmune2D.jl")

function runSimulation(vi2D, mParams, simSet, sampDists, time, speciationTimes, extinctionTimes, nonLocalTimes)
    extinctionFlag::Bool = false

    for t in time
        extinctionFlag, nonLocalFlag = simulationStep!(vi2D, mParams, simSet)

        extinctionFlag && (println("WARNING: virus extinct"); push!(extinctionTimes, t); break)
        nonLocalFlag && (push!(nonLocalTimes, t); println("Non-local event!"))

        if isTime2Sample(t, simSet)
            println("t = $t")
            sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
            hasSpeciationHappened(sampDists) && push!(speciationTimes, t)
        end
    end

    return extinctionFlag
end

r = 50
R0 = 1.5
Nh::Int = 5e8
mu = 0.12
nonLocalMutProb = 0
nonLocalJump = 40
localMutationKernel = NextNeighbour()
mutationKernel = piecewiseKernel2D(nonLocalMutProb, nonLocalJump, localMutationKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 250
dt = 0.1
dtSampling = 4
simSet = simulationConfig(tmax, dt, dtSampling)
sampDists = sampledResults()

x = -150:150
y = -150:150

vi2D = viralImmuneDistribution2D(x, y)
getInitialDistribution!(vi2D, mParams, sampDists)

nRuns = 4
speciationTimes = Float64[]
extinctionTimes = Float64[]
nonLocalTimes = Float64[]

println("Starting simulation!")

for run in 1:nRuns
    println("Run $run of $nRuns")

    time = dt:dt:tmax
    extinctionFlag = runSimulation(vi2D, mParams, simSet, sampDists, time, speciationTimes, extinctionTimes, nonLocalTimes)

    extinctionFlag ?
    getInitialDistribution!(vi2D, mParams, sampDists) :
    isolateBestSpeciesBack2Origin!(vi2D, mParams)
end


# plotAnimation(sampDists, vi2D.x, vi2D.y)