include("../../../code/mutantGrowth/secondMutantStudy.jl")

r = 40
R0 = 1.3
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

(Nt, xt, sigmat, uTt, absorbedState, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

tTransient = 50
idxTransient = findfirst(t .== tTransient)
if absorbedState == 0
    vAv = (xt[end] - xt[tTransient]) / (t[end] - t[tTransient])
end
uTAv = mean(uTt[tTransient:end])

establishmentEvents = 20
deltaRGrid = 0:0.05:2
nEstablished = Vector{Float64}(undef, length(deltaRGrid))
nSimulated = Vector{Float64}(undef, length(deltaRGrid))

xEstablishment = 0.05

for i in eachindex(deltaRGrid)

    println("=============================================")
    println("Starting search for deltaR = $(deltaRGrid[i])")
    println(".\n.\n.\n.\n.\n.")

    deltaRMutant =  deltaRGrid[i]
    duMutant = deltaRMutant * uTAv

    nSimulated[i], nEstablished[i], searches = countMutantTillNEst(nxBack0, hxBack0, duMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x, establishmentEvents, maxSearches = 40, maxItrFixation = 300, xEstab = xEstablishment)
end

establishmentProb = nEstablished ./ nSimulated
sGrid = s .* uTAv .* deltaRGrid
sigma = sqrt.((nEstablished .+ 1) .* (nSimulated .- nEstablished .+1) ./ ((nSimulated .+ 2).^2 .* (nSimulated .+ 3)))
sigma = [max(sigma[i], 1 / nSimulated[i]) for i in eachindex(sigma)]

plotConfig()
p = plot(deltaRGrid, sGrid ./(1 .+ sGrid), colour = :black, label = "Haldane-like limit", title = raw"Establishment probability for $r = 40, R_0=1.3, D =" * "$(round(D, sigdigits=2))" * raw",$" * "\n" * raw"and with $\langle{v}\rangle =" * "$(round(vAv, sigdigits=2))" * raw", \langle{u_T}\rangle =" * "$(round(uTAv,sigdigits=2))" * raw", x_\mathrm{est} = " * "$xEstablishment" * raw"$", titlefontsize = 14, topmargin = 10Plots.pt)
plot!(p, deltaRGrid, establishmentProb, xlabel = raw"$u/u_T$", ylabel = "Establishment probability", yerror = sigma, colour = :steelblue, lc = :steelblue, mc = :steelblue, label = "Data")
savefig(p, "figures/mutantGrowth/EstablishmentProbabilityxEst$(xEstablishment).png")

vAvVect = [vAv; zeros(length(deltaRGrid)-1)]
uTAvVect = [uTAv; zeros(length(deltaRGrid) -1)]
survProbTable = Tables.table([vAvVect uTAvVect deltaRGrid sGrid nEstablished nSimulated establishmentProb sigma], header = ["vAv", "uTAv", "uM/uT", "sM", "nEstablished", "nSimulated", "establishmentProbability", "sigma(estProb)"])

baseFolder = "../../../simulations/mutantGrowth/"
fileName = "establishmentProbability_r$(r)R0$(R0)D$(D)xEst$(xEstablishment).csv"
CSV.write(joinpath(baseFolder, fileName), survProbTable)