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

r = 10
R0 = 1.3
Nh::Int = 1e8
mu = 0.10
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

N0 = 10
vi2D = viralImmuneDistribution2D(x, y)
vi2D.nx[151, 151] = 10
getGrowthRate!(vi2D, mParams)
sampleDistributionsNSpecies!(vi2D, sampDists, mParams)

nRuns = 2
speciationTimes = Float64[]
extinctionTimes = Float64[]
nonLocalTimes = Float64[]

println("Starting simulation!")

for run in 1:nRuns
    println("Run $run of $nRuns")

    time = dt:dt:tmax
    extinctionFlag = runSimulation(vi2D, mParams, simSet, sampDists, time, speciationTimes, extinctionTimes, nonLocalTimes)

    extinctionFlag && begin
        global vi2D = viralImmuneDistribution2D(x, y)
        vi2D.nx[151, 151] = 10
        getGrowthRate!(vi2D, mParams)
        sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
    end
end


# plotAnimation(sampDists, vi2D.x, vi2D.y)