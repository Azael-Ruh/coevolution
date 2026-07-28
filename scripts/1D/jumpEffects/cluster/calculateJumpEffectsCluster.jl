using JLD2

include(expanduser("~/coevolution/code/mutantGrowth/secondMutantStudy.jl"))

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
s = log(R0)/r

mutationRate = parse(Float64, ARGS[3])
mutationKernel = eval(Meta.parse(ARGS[4])) # Dangerous
localKernel = eval(Meta.parse(ARGS[5])) # Dangerous
mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2

Nh = parse(Int, ARGS[6])

tmax = 6000

vFKPP = 2 * sqrt((R0 - 1) * D)
xmax = 6*max(500, round(Int, vFKPP*tmax + vFKPP^2/(D*s)))

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

dt = 0.1
dtSampling = 1
t = 0:dtSampling:tmax

(Nt, xt, sigmat, uTt, absorbedState, _, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, localKernel, dt, tmax, dtSampling, x)

tTransient = 50
idxTransient = findfirst(t .>= tTransient)
if absorbedState != 0
    println("xmax = $xmax")
    println("xt = $xt")
    println("Nt = $Nt")
    error("Mutant absorbed on base simulation!")
end
vAv = (xt[end] - xt[idxTransient]) / (t[end] - t[idxTransient])
NAv = mean(Nt[idxTransient:end])
uTAv = mean(uTt[idxTransient:end])

tmax = parse(Float64, ARGS[7])
t = 0:dtSampling:tmax
tTransient = 50
idxTransient = findfirst(t .>= tTransient)

totalRuns = parse(Int, ARGS[8])

println("=============================================")
println("Starting simulation for Delta = $(mutationKernel.nonLocalJump)")
println(".\n.\n.\n.\n.\n.")

vMod = 0
NMod = 0
survivedRuns = 0
nRuns = 0
while survivedRuns < totalRuns
    println("run $survivedRuns")
    (NtJump, xtJump, sigmatJump, uTtJump, absorbedStateJump, idxAbsorbedJump, _, _) = simulateWaveMacro(nxBack0, hxBack0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
    newV = (xtJump[end] - xtJump[idxTransient]) / (t[end] - t[idxTransient])
    newN = mean(NtJump[idxTransient:end])
    println("Found speed: newV = $newV")
    println("Found average size: newN = $newN")
    global vMod += (absorbedStateJump == 0 ? newV : 0)
    global NMod += (absorbedStateJump == 0 ? newN : 0)
    global survivedRuns += 1*(absorbedStateJump == 0)
    global nRuns += 1
end

if survivedRuns > 0
    vMod = vMod / survivedRuns
    NMod = NMod / survivedRuns
    println("Speed before = $vAv, new speed = $vMod, size before = $NAv, new size = $NMod, calculated through $survivedRuns survived simulations.")
    
    saveDir = expanduser("~/coevolution/simulations/jumpEffects")
    saveFile = "jumpEffects_r$(r)R0$(R0)D$(round(D, sigdigits = 2))tmax$(tmax)totalRuns$(totalRuns)Delta$(mutationKernel.nonLocalJump).jld2"
    jldsave(joinpath(saveDir, saveFile); vMod, NMod, survivedRuns, nRuns)
else
    println("ERROR: No survived runs found :(")
end