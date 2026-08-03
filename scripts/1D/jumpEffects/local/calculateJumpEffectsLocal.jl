include(expanduser("~/PhDVirusImmuneCoEvolution/coevolution/code/mutantGrowth/secondMutantStudy.jl"))

r = 40
R0 = 1.5
s = log(R0)/r

mutationRate = 0.2
nonLocalMutProb = 1e-6
nonLocalJumpVect = [0, 10, 20, 30, 40, 50, 60, 70]
localKernel = Normal(0,1)
mutationScale = std(localKernel)*(1-nonLocalMutProb)
D = mutationRate * mutationScale^2 / 2

Nh = 10000000

tmax = 8000
vFKPP = 2 * sqrt((R0 - 1) * D)
xmax = 4*max(500, round(Int, vFKPP*tmax + vFKPP^2/(D*s)))

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, localKernel, Nh, xmax)

dt = 0.1
dtSampling = 1
t = 0:dtSampling:tmax

(Nt, xt, sigmat, uTt, absorbedState, _, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, localKernel, dt, tmax, dtSampling, x)

tTransient = 100
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

tmax = 1000
t = 0:dtSampling:tmax
tTransient = 50
idxTransient = findfirst(t .>= tTransient)

totalRuns = 5
vModVect = Vector{Float64}(undef, length(nonLocalJumpVect))
NModVect = Vector{Float64}(undef, length(nonLocalJumpVect))
survivedRunsVect = Vector{Float64}(undef, length(nonLocalJumpVect))

for i in eachindex(nonLocalJumpVect)

    mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJumpVect[i], localKernel)
    println("=============================================")
    println("Starting simulation for Delta = $(mutationKernel.nonLocalJump)")
    println(".\n.\n.\n.\n.\n.")

    vMod = 0
    NMod = 0
    survivedRuns = 0
    while survivedRuns < totalRuns
        println("run $survivedRuns")
        (NtJump, xtJump, sigmatJump, uTtJump, absorbedStateJump, idxAbsorbedJump, _, _) = simulateWaveMacro(nxBack0, hxBack0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
        newV = (xtJump[end] - xtJump[idxTransient]) / (t[end] - t[idxTransient])
        newN = mean(NtJump[idxTransient:end])
        println("Found speed: newV = $newV")
        println("Found average size: newN = $newN")
        vMod += (absorbedStateJump == 0 ? newV : 0)
        NMod += (absorbedStateJump == 0 ? newN : 0)
        survivedRuns += 1*(absorbedStateJump == 0)
    end

    if survivedRuns > 0
        vModVect[i] = vMod / survivedRuns
        NModVect[i] = NMod / survivedRuns
        survivedRunsVect[i] = survivedRuns
        println("Speed before = $vAv, new speed = $(vModVect[i]), size before = $NAv, new size = $(NModVect[i]), calculated through $survivedRuns survived simulations.")
    else
        println("ERROR: No survived runs found :(")
    end
end

plotConfig()
scatter(nonLocalJumpVect, vModVect, xlabel = raw"$\Delta$", ylabel = raw"$\Delta v$")