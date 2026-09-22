include("../src/viralImmune2D.jl")

using JLD2

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
Nh::Int = parse(Int, ARGS[3])
mu = parse(Float64, ARGS[4])
nonLocalMutProb = parse(Float64, ARGS[5]) # Dangerous
nonLocalJump = parse(Float64, ARGS[6])
localMutationKernel = NextNeighbour()
mutationKernel = piecewiseKernel2D(nonLocalMutProb, nonLocalJump, localMutationKernel)
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = parse(Float64, ARGS[7])
dt = 0.1
dtSampling = 2
simSet = simulationConfig(tmax, dt, dtSampling)
sampDists = sampledResults()

x = -150:250
y = -250:250

vi2D = viralImmuneDistribution2D(x, y)
getInitialDistribution!(vi2D, mParams, sampDists)

nCycles = parse(Int, ARGS[8])
speciationTimes = Float64[]
extinctionTimes = Float64[]
nonLocalTimes = Float64[]

println("Starting simulation!")

for cycle in 1:nCycles 
    println("Cycle $cycle of $nCycles")

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
        end
    end

    if extinctionFlag
        push!(extinctionTimes, t)
        getInitialDistribution!(vi2D, mParams, sampDists)
    else
        isolateBestSpeciesBack2Origin!(vi2D, mParams)
    end
    
    global sampDists = sampledResults()
    sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
end

nSpeciationEvents = length(speciationTimes) #TODO: improve for speciation detection!
nExtinctionEvents = length(extinctionTimes)
nNonLocalEvents = length(nonLocalTimes)

run = parse(Int, ARGS[9])

saveDir = expanduser("~/coevolution/simulations/2D/speciationNExtinction")
saveFile = "speciationExtinction_r$(r)R0$(R0)mu$(mu)tmax$(tmax)nCycles$(nCycles)nonLocalProb$(nonLocalMutProb)Delta$(mutationKernel.nonLocalJump)_run$(run).jld2"
jldsave(joinpath(saveDir, saveFile); nSpeciationEvents, speciationTimes, nExtinctionEvents, extinctionTimes, nNonLocalEvents, nonLocalTimes)