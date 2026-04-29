using Plots, SpecialFunctions, NLsolve, LinearAlgebra, CSV, LsqFit
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

# Simulation parameters
Nh = 10000000
r = 40
R0 = 1.3
s = log(R0)/r

mutationRate = 0.5
mutationKernel = Normal(0, 1)
mutationScale = std(mutationKernel)
D = mutationRate*mutationScale^2/2

tmax = 500
dt = 0.1
dtSampling = 1
t = 0:dtSampling:tmax
tTransient = 50

(NAv, Nstd, vAv, sigmaAv, uTAv) = simulateWaveStatisticsFull(R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, tTransient; xmax = 1000, s = log(R0)/r, D = mutationRate*mutationScale^2/2, initialCond = "steadyState")

# Fitness space paremeters

zeta = vAv^2/4D
vs = vAv * s
Ds = D * s^2
d = Ds^(1/3)

# Numerical solution
function discretisedODE!(f, r, w, dr)
    f[1] = w[1]
    f[end] = w[end] - r[end]/(1 + r[end])
    f[2:end-1] .= vs/2dr .* (w[3:end] - w[1:end-2]) .- r[2:end-1] .* w[2:end-1] .+ (1 .+ r[2:end-1]) .* w[2:end-1].^2 .- Ds/dr^2 .* (w[3:end] .- 2 .* w[2:end-1] .+ w[1:end-2])
end

dr = 5e-4
rT = s*uTAv
rVect = -10rT:dr:10rT

# Initial guess
alphac(rc) = 2zeta - vs/d*airyaiprime((zeta - rc)/d)/airyai((zeta - rc)/d)
wc(rc) = (rc-alphac(rc))/(1+rc)
Ac(rc) = wc(rc)/(exp(vs*rc/(2Ds))*airyai((zeta - rc)/d))
Cc(rc) = vs * exp(rc^2/2vs)/wc(rc) - (vs * exp(rc^2/2vs) + sqrt(pi * vs / 2) * erfi(rc / sqrt(2vs)))
whighGood(r, rc) = vs * exp(r^2 / 2vs) / (vs * exp(r^2 /2vs) + Cc(rc) + sqrt(pi * vs / 2) * erfi(r / sqrt(2vs)))
wlowGood(r, rc) = Ac(rc) * airyai((zeta-r)/d) * exp(vs*r/2Ds)
wtheo(r, rc) = whighGood.(r,rc) .* (r .> rc) .+ wlowGood.(r,rc) .* (r .< rc)
rc0 = zeta + 1.6d
w0 = wtheo(rVect, rc0)
w0[1] = 0

# Solution
discretisedODEToSolve! = (f,w) -> discretisedODE!(f, rVect, w, dr)
sol = nlsolve(discretisedODEToSolve!, w0)
wSol = sol.zero

# Fitting the best rc
fittingFunc(r, p) = wtheo(r, p[1])
fit = curve_fit(fittingFunc, rVect, wSol, [rc0])
rcFit = fit.param[1]
xiFit = (rcFit - zeta)/d

# Plot it
plotConfig()
p = plot(rVect, wSol, xlims = (0, 0.3), ylims = (0, 0.3), colour = :coral, lw = 1, label = "Numerical solution", title = raw"$D_s =" * "$(round(Ds, sigdigits = 2))" * ", v_s =" * "$(round(vs, sigdigits = 2))" * raw", \zeta =" * "$(round(zeta, sigdigits = 2))" * raw", d =" * "$(round(d, sigdigits = 2))" * raw", \xi_c =" * "$(round(xiFit, sigdigits = 3))" * raw"$", xlabel = raw"$r$", ylabel = raw"Fixation probability $w(r)$", top_margin = 10Plots.pt)
plot!(p, rVect, wtheo.(rVect, rcFit), colour = :black, label = "Best fit")
plot!(p, rVect, rVect./(1 .+rVect), colour = :black, ls = :dash, label = "Haldane limit")
vline!(p, [rcFit], c = :black, ls = :dashdot, label = raw"Matching point $r_c$")

# Bulk wave
bulkWave(r) = airyai((zeta-r)/d) * exp(-vs*r/2Ds)

#Data 
xEstablishment = 0.05
baseFolder = "simulations/mutantGrowth/"
println(pwd())
fileName = "establishmentProbability_r$(r)R0$(R0)D$(D)xEst$(xEstablishment).csv"
(vAvVect, uTAvVect, deltaRGrid, sGrid, nEstablished, nSimulated, establishmentProb, sigma) = Vector.(eachcol(CSV.read(joinpath(baseFolder, fileName), CSV.Tables.matrix)))



rPlot = rVect[findfirst(rVect .>= -0.5rT): findfirst(rVect .>= 2rT)]
q = scatter(deltaRGrid, establishmentProb, yerror = sigma, colour = :steelblue, lc = :steelblue, mc = :steelblue, ms = 5, msc = :steelblue, msw = 2, label = "Data")
plot!(q, rPlot./rT, wSol[findfirst(rVect .>= -0.5rT): findfirst(rVect .>= 2rT)], xlims = (-0.5, 2), ylims = (0, 0.3), colour = :coral, lw = 2, xlabel = raw"$r$", ylabel = raw"Fixation probability $w(r)$", top_margin = 10Plots.pt, label = "Numerics")
plot!(q, rPlot./rT, wtheo.(rPlot, rcFit), colour = :black, label = "Asymptotics", lw = 2)
plot!(q, (-0.5rT:dr:1rT)./rT, bulkWave.(-0.5*rT:dr:1rT)./bulkWave(0).*0.19, c = :red, lw = 2, label = "Wave")
plot!(q, rPlot./rT, rPlot./(1 .+ rPlot), lw = 1.5, ls = :dash, c = :black, label = "Haldane limit")
vspan!(q, [-1, 1], color = :coral, alpha = 0.15, label = "Viral span")
savefig(q, "figures/mutantGrowth/EstablishmentProbExample.svg")