include("../../../../../code/mutantGrowth/secondMutantStudy.jl")

r = 40
R0 = 1.2
s = log(R0)/r

mutationRate = 0.2
mutationKernel = Normal(0, 1)
mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2

tmax = 2000

vFKPP = 2 * sqrt((R0 - 1) * D)
xmax = 2*max(500, round(Int, vFKPP*tmax + vFKPP^2/(D*s)))

Nh = 10000000
(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

dt = 0.1
dtSampling = 1
t = 0:dtSampling:tmax

(Nt, xt, sigmat, uTt, absorbedState, _, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

tTransient = 50
idxTransient = findfirst(t .== tTransient)
if absorbedState == 0
    vAv = (xt[end] - xt[idxTransient]) / (t[end] - t[idxTransient])
end
uTAv = mean(uTt[idxTransient:end])

tmax = 500
deltaRGrid = 0:0.05:2
totalRuns = 20
tExtinction = Array{Float64, 2}(undef, length(deltaRGrid), totalRuns)

for i in eachindex(deltaRGrid)

    println("=============================================")
    println("Starting search for deltaR = $(deltaRGrid[i])")
    println(".\n.\n.\n.\n.\n.")

    deltaRMutant =  deltaRGrid[i]
    duMutant = deltaRMutant * uTAv

    tExtinction[i,:] = produceMutantExtinctionTimes(nxBack0, hxBack0, duMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x, totalRuns)
end

nEstablished = [sum(tExtinction[i,:] .== tmax) for i in eachindex(deltaRGrid)]
nSimulated = totalRuns*ones(length(deltaRGrid))
establishmentProb = nEstablished ./ nSimulated
sGrid = s .* uTAv .* deltaRGrid
sigma = sqrt.((nEstablished .+ 1) .* (nSimulated .- nEstablished .+1) ./ ((nSimulated .+ 2).^2 .* (nSimulated .+ 3)))
sigma = [max(sigma[i], 1 / nSimulated[i]) for i in eachindex(sigma)]

plotConfig()
p = plot(deltaRGrid, sGrid ./(1 .+ sGrid), colour = :black, label = "Haldane-like limit", title = raw"Survival probability for $r = 40, R_0=1.3, D =" * "$(round(D, sigdigits=2))" * raw",$" * "\n" * raw"and with $\langle{v}\rangle =" * "$(round(vAv, sigdigits=2))" * raw", \langle{u_T}\rangle =" * "$(round(uTAv,sigdigits=2))" * raw", t_\mathrm{max} = " * "$tmax" * raw"$", titlefontsize = 14, topmargin = 10Plots.pt)
plot!(p, deltaRGrid, establishmentProb, xlabel = raw"$u/u_T$", ylabel = "Establishment probability", yerror = sigma, colour = :steelblue, lc = :steelblue, mc = :steelblue, label = "Data")
# savefig(p, "figures/mutantGrowth/EstablishmentProbabilityxEst$(xEstablishment).png")

vAvVect = [vAv; zeros(length(deltaRGrid)-1)]
uTAvVect = [uTAv; zeros(length(deltaRGrid) -1)]
survProbTable = Tables.table([vAvVect uTAvVect deltaRGrid sGrid nEstablished nSimulated establishmentProb sigma], header = ["vAv", "uTAv", "uM/uT", "sM", "nEstablished", "nSimulated", "establishmentProbability", "sigma(estProb)"])

baseFolder = "simulations/mutantGrowth/"
fileName = "establishmentProbability_r$(r)R0$(R0)D$(D)xEstFulltmax$(tmax).csv"
CSV.write(joinpath(baseFolder, fileName), survProbTable)

timeSurvivalTable = Tables.table([deltaRGrid tExtinction], header = ["deltaR"; ["run $(i)" for i in 1:totalRuns]])
baseFolder = "simulations/mutantGrowth/timeSurvivalProb"
fileName = "extinctionTimes_r$(r)R0$(R0)D$(D)tmax$(tmax)totalRuns$(totalRuns).csv"
CSV.write(joinpath(baseFolder, fileName), timeSurvivalTable)