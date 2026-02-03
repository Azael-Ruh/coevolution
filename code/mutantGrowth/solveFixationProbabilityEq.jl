using Plots, SpecialFunctions, NLsolve, LinearAlgebra, CSV
include("../../code/simulate/coevolution1DSimulationTools.jl")

# FINITE DIFFERENCES FOR THE SIMULATED CASE
r = 40
R0 = 1.3
s = log(R0)/r
D = 0.25
xEstablishment = 0.05

# Data 
baseFolder = "simulations/mutantGrowth/"
fileName = "establishmentProbability_r$(r)R0$(R0)D$(D)xEst$(xEstablishment).csv"
vAv, uTAv, deltaRGrid, sGrid, nEstablished, nSimulated, establishmentProb, sigma = Vector.(eachcol(CSV.read(joinpath(baseFolder, fileName), CSV.Tables.matrix)))
vAv = vAv[1]
uT = uTAv[1]

xi = vAv^2/D
zeta = xi/4
vs = vAv * s
Ds = D * s^2
d = Ds^(1/3)

# Numerical solution
function fnEquation(r, w, dr)
    fw = zero(w)
    fw[1] = w[1]
    fw[end] = w[end] - r[end]/(1 + r[end])
    fw[2:end-1] .= vs/2dr .* (w[3:end] - w[1:end-2]) .- r[2:end-1] .* w[2:end-1] .+ (1 .+ r[2:end-1]) .* w[2:end-1].^2 .- Ds/dr^2 .* (w[3:end] .- 2 .* w[2:end-1] .+ w[1:end-2])
    return fw
end
function fnEquation!(f, r, w, dr)
    f[1] = w[1]
    f[end] = w[end] - r[end]/(1 + r[end])
    f[2:end-1] .= vs/2dr .* (w[3:end] - w[1:end-2]) .- r[2:end-1] .* w[2:end-1] .+ (1 .+ r[2:end-1]) .* w[2:end-1].^2 .- Ds/dr^2 .* (w[3:end] .- 2 .* w[2:end-1] .+ w[1:end-2])
end

dr = 5e-4
rT = s*uT
rVect = -10rT:dr:10rT

rc = zeta + 1.9d
wlow = r -> rc/(1+rc)*airyai((zeta-r)/d)/airyai((zeta-rc)/d) * exp(vs*(r - rc)/2Ds)
whigh = r -> r/(1+r)
# Analytical first asymptotic expression
w0 = [wlow.(rVect[rVect .< rc]); whigh.(rVect[rVect .>= rc])]
w0[1] = 0

fToSolve! = (f,w) -> fnEquation!(f, rVect, w, dr)
sol = nlsolve(fToSolve!, w0)
wSolv = sol.zero

# Better analytical solution
rc = zeta + 1.6d
alphac = 2zeta - vs/d*airyaiprime((zeta - rc)/d)/airyai((zeta - rc)/d)
wc = (rc-alphac)/(1+rc)
A = wc/(exp(vs*rc/(2Ds))*airyai((zeta - rc)/d))
C = vs * exp(rc^2/2vs)/wc - (vs * exp(rc^2/2vs) + sqrt(pi * vs / 2) * erfi(rc / sqrt(2vs)))

whighGood = r -> vs * exp(r^2 / 2vs) / (vs * exp(r^2 /2vs) + C + sqrt(pi * vs / 2) * erfi(r / sqrt(2vs)))
wlowGood = r -> wc*airyai((zeta-r)/d)/airyai((zeta-rc)/d) * exp(vs*(r - rc)/2Ds)
wtheo = [wlowGood.(rVect[rVect .< rc]); whighGood.(rVect[rVect .>= rc])]

rTtheo = vs^2/4Ds + 2.3381d - 2Ds/vs

plotConfig()
p = plot(rVect./rT, wSolv, xlims = (-0.5,2), colour = :coral, lw = 1, label = "Numerical solution", ylims = (0, 0.3))
plot!(p, rVect./rT, w0, colour = :black, ls = :dash, label = "Asymptotic solution")
plot!(p, rVect./rT, wtheo, colour = :black, label = "Improved asymptotic solution")
vline!(p, [rc/rT], lw=0.5, colour = :black, ls = :dashdot, label = raw"Matching point $r_c$")
scatter!(p, deltaRGrid, establishmentProb, xlabel = raw"$r/su_T$", ylabel = "Establishment probability", yerror = sigma,  mc = :steelblue, msc = :steelblue, lc = :steelblue, label = "Data")

fwave = r -> airyai((zeta-r)/d)/airyai((zeta-rc)/d) * exp(-vs*(r-rc)/2Ds)
nwave = [fwave.(rVect[rVect .< rc]); zero(rVect[rVect .> rc])]
nwave = nwave ./ maximum(nwave) .* rc/(1+rc)
plot!(rVect./rT, nwave, c = :firebrick1, label = "Diffusive wave")
vspan!(p, [-rTtheo, rTtheo]./rT, color = :coral, alpha = 0.15, label = "Theoretical viral span")
savefig(p, "figures/mutantGrowth/EstablishmentProbabilityxEst$(xEstablishment).png")