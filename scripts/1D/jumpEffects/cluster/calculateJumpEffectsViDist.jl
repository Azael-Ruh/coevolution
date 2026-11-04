include("../src/viralImmuneDistribution.jl")

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

function getMacroscopicAndExtinction(x, mParams, simSet)

    sampMacro = macroscopicSampledResults()
    sampDists = sampledResults()
    time = 0:simSet.dt:(simSet.tmax-simSet.dt)

    viDist = viralImmuneDistribution(x)
    getInitialDistribution!(viDist, mParams, sampMacro)

    survivalFlag = true
    for t in time
        survivalFlag, _ = simulationStep!(viDist, mParams, simSet)
        survivalFlag || (println("WARNING: virus extinct."); break)

        isTooCloseToBoundary(viDist, mParams) && (translateDistributionBackLeft!(viDist, mParams); println("Translated distribution left!"))
        isTime2Sample(t, simSet) && (sampleMacroscopicObservables!(viDist, sampMacro); sampleDistributions!(viDist, sampDists))
    end

    return sampMacro, survivalFlag
end

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
Nh::Int = parse(Int, ARGS[6])
mu = parse(Float64, ARGS[3])
mutationKernel = eval(Meta.parse(ARGS[5]))
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = 6000
dt = 0.02
dtSampling = 1
simSet = simulationConfig(tmax, dt, dtSampling)

xmax = 500
x = round(Int, -5*r):xmax

sampMacro, sampDists, _ = getSimulatedMacroscopicObservables(x, mParams, simSet)
vAv = (sampMacro.xt[end] - sampMacro.xt[1]) ./ tmax
NAv = mean(sampMacro.Nt)
uTAv = mean(sampMacro.xTt .- sampMacro.xt)

println("========== Original simulation results ==========")
println("Found average speed vAv  = $vAv")
println("Found average size  NAv  = $NAv")
println("Found average tip   uTAv = $uTAv")
println("=================================================")

mutationKernel = eval(Meta.parse(ARGS[4]))
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

tmax = parse(Float64, ARGS[7])
tTransient = 100
idxTransient = round(Int, tTransient ./ dtSampling) + 1
simSet = simulationConfig(tmax, dt, dtSampling)

totalRuns = parse(Int, ARGS[8])
println("=============================================")
println("Starting jumpy simulation for Delta = $(mutationKernel.nonLocalJump)")
println(".\n.\n.\n.\n.\n.")

survivedRuns = 0
vMod = 0
NMod = 0
nRuns = 0
while survivedRuns < totalRuns
    println("run $survivedRuns")
    local sampMacro, survivalFlag = getMacroscopicAndExtinction(x, mParams, simSet)
    if survivalFlag
        newV = (sampMacro.xt[end] - sampMacro.xt[idxTransient]) / (tmax - tTransient)
        newN = mean(sampMacro.Nt[idxTransient:end])
        println("Found speed: newV = $newV")
        println("Found average size: newN = $newN")

        global vMod += newV
        global NMod += newN
        global survivedRuns += 1
    end
    global nRuns += 1
end


if survivedRuns > 0
    vMod = vMod / survivedRuns
    NMod = NMod / survivedRuns
    println("Speed before = $vAv, new speed = $vMod, size before = $NAv, new size = $NMod, calculated through $survivedRuns survived simulations.")

    saveDir = expanduser("~/coevolution/simulations/jumpEffects")
    saveFile = "jumpEffects_r$(r)R0$(R0)D$(round(mParams.D, sigdigits = 2))tmax$(tmax)totalRuns$(totalRuns)Delta$(mutationKernel.nonLocalJump).jld2"
    jldsave(joinpath(saveDir, saveFile); vMod, NMod, survivedRuns, nRuns)
else
    println("ERROR: No survived runs found :(")
end