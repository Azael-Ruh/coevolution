include("src/disorderedImmuneDistributions.jl")


function runSimulation!(vi2D, mParams, simSet, sampDists, time)
    extinctionFlag::Bool = false

    for t in time
        extinctionFlag, _ = simulationStep!(vi2D, mParams, simSet)

        extinctionFlag && (println("WARNING: virus extinct"); break)

        if isTime2Sample(t, simSet)
            println("t = $t")
            sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
        end
    end

    return extinctionFlag
end

r = 20
originalR0 = 2.7
targetR0 = 1.4
Nh::Int = 1e7
mu = 0.10
mutationKernel = NextNeighbour()
mParams = modelParams(r, originalR0, Nh, mu, mutationKernel)

tmax = 250
dt = 0.1
dtSampling = 10
simSet = simulationConfig(tmax, dt, dtSampling)
sampDists = sampledResults()

x = -400:400
y = -400:400

vi2D = viralImmuneDistribution2D(x, y)
landsize=(length(x), length(y))
Reff, hx = getRandomImmuneLandscape(originalR0, targetR0, landsize, mParams.Hkernel, mParams.Nh)
vi2D.hx = hx
vi2D.Reff = Reff

nRuns = 6

while 1 != 0

    global sampDists = sampledResults()
    vi2D.nx = zero(hx)
    randIdx = rand(CartesianIndices((100:(length(x)-100), 100:(length(y)-100))))
    println("Initial randomly selected growth rate $(Reff[randIdx])")
    vi2D.nx[randIdx] = 10
    sampleDistributionsNSpecies!(vi2D, sampDists, mParams)
    extinctionFlag = false

    println("Starting simulation!")
    for run in 1:nRuns
        println("Run $run of $nRuns")

        time = dt:dt:tmax
        extinctionFlag = runSimulation!(vi2D, mParams, simSet, sampDists, time)
        extinctionFlag && break
    end

    extinctionFlag || break
end