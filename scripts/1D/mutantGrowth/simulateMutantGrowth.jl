include("../../../code/mutantGrowth/secondMutantStudy.jl")

r = 60
R0 = 2
s = log(R0)/r

mutationRate = 0.5
mutationKernel = Normal(0, 1)
mutationScale = std(mutationKernel)
D = mutationRate*mutationScale^2/2

tmax = 500

vFKPP = 2 * sqrt((R0 - 1) * D)
xmax = 2*max(500, round(Int, vFKPP*tmax + vFKPP^2/D))

Nh = 10000000
(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

dt = 0.1
dtSampling = 1
t = 0:dtSampling:tmax

(Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

tTransient = 50
idxTransient = findfirst(t .== tTransient)
if absorbedState == 0
    vAv = (xt[end] - x[tTransient]) / (t[end] - t[tTransient])
end
uTAv = mean(uTt[tTransient:end])

deltaRMutant =  1.1
duMutant = deltaRMutant * uTAv

(nxBackground, nxMutant, hx, extinctionState, establishmentState) = simulateMutantTillEstablishment(nxBack0, hxBack0, duMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x, maxItr = 4000)

plotConfig()
if !extinctionState || establishmentState
    g = animateSimulationMutant(nxBackground, nxMutant, hx, x, Nh)
    NtBackground = dropdims(sum(nxBackground, dims = 2), dims = 2)
    NtMutant = dropdims(sum(nxMutant, dims = 2), dims = 2)
    p = plotMutantSweep(NtBackground, NtMutant, t)
end