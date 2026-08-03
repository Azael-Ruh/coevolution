include(expanduser("~/PhDVirusImmuneCoEvolution/coevolution/code/mutantGrowth/secondMutantStudy.jl"))

r = 40
R0 = 1.2
s = log(R0)/r
mutationRate = 0.10
mutationKernel = Normal(0,1)
mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2
Nh = 10000000
tmax = 500.0
totalRuns = 4000

xmax = 2000
dt = 0.1
t = 0:dt:tmax
dtSampling = 2
(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)
(Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxLoc, hxLoc) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
uTAv = mean(uTt[50:end])
vAv = (xt[end] - xt[50]) ./ (t[end] .- (50-1)*dtSampling)

baseFolder = expanduser("~/PhDVirusImmuneCoEvolution/coevolution/simulations/mutantGrowth/timeSurvivalProb")
fileName = "extinctionTimes_r$(r)R0$(R0)D$(D)tmax$(tmax)totalRuns$(totalRuns).csv"
tExtTable = CSV.read(joinpath(baseFolder, fileName), CSV.Tables.matrix)

deltaR = tExtTable[:, 1]
tExt = tExtTable[:,2:end]

dt = 0.1
t = 0:dt:tmax
survivalProb = [sum(tExt[i,:] .>= t[j]) for i in eachindex(deltaR), j in eachindex(t)] ./ totalRuns

w = -log.(1 .- survivalProb)
wTheoSelection = (r, tau) -> (r == 0 ? 1/tau : r/(1-exp(-r*tau)))
vs = vAv * s
wTheoBulk = (r, tau) -> (tau == 0 ? Inf : vs*exp(r^2/2vs)/ sum([exp(a^2/2vs) for a in (r-vs*tau):1e-5:r] .* 1e-5))
rVect = s.*uTAv.*deltaR
plotConfig()
animation = @animate for i in eachindex(t)[1:10:end]
    p = scatter(rVect, w[:,i], mc=:steelblue, ms = 5, ylabel=raw"$-\log(1-\mathbb{P}_\mathcal{S}(x_0, t))$", xlabel=raw"$sx_0$", label = "Data")
    plot!(p, rVect, wTheoSelection.(rVect, t[i]), c = :black, label = "Only fitness solution")
    plot!(p, rVect, wTheoBulk.(rVect, t[i]), c = :black, ls = :dash, label = "No diffusion solution")
end 
g = gif(animation)