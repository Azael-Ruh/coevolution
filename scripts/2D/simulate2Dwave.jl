include("./src/viralImmune2D.jl")

r = 30
R0 = 2.5
Nh::Int = 1e7
mu = 0.10
nonLocalMutProb = 2e-6
nonLocalJump = 30
localMutationKernel = NextNeighbour()
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

println("Starting simulation!")

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


# plotAnimation(sampDists, vi2D.x, vi2D.y)