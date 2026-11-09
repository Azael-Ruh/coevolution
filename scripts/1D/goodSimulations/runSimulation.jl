include("src/viralImmuneDistribution.jl")

function getSimulatedMacroscopicObservables(x, mParams, simSet)

    sampMacro = macroscopicSampledResults()
    sampDists = sampledResults()
    time = 0:simSet.dt:(simSet.tmax-simSet.dt)

    survivalFlag = false
    while !survivalFlag

        viDist = viralImmuneDistribution(x)
        getInitialDistribution!(viDist, mParams, sampMacro)

        for t in time
            survivalFlag, _ = simulationStep!(viDist, mParams, simSet)
            survivalFlag || (println("WARNING: virus extinct. Trying again"); break)

            isTooCloseToBoundary(viDist, mParams) && (translateDistributionBackLeft!(viDist, mParams); println("Translated distribution left!"))
            isTime2Sample(t, simSet) && (sampleMacroscopicObservables!(viDist, sampMacro); sampleDistributions!(viDist, sampDists))
        end
    end

    return sampMacro, sampDists, survivalFlag
end

r = 40
R0 = 1.2
Nh::Int = 1e8
mu = 0.4
mutationKernel = piecewiseKernel(1e-6, 15, Normal(0, 1))
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 6000
dt = 0.02
dtSampling = 1
simSet = simulationConfig(tmax, dt, dtSampling)

xmax = 500
x = round(Int, -5*r):xmax

sampMacro, sampDists, _ = getSimulatedMacroscopicObservables(x, mParams, simSet)
t = 0:dtSampling:tmax
p = plot(t, sampMacro.xTt, lw=2, colour=:coral)
