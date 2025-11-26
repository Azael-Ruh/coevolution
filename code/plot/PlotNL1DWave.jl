using Plots, Interpolations, Tables, CSV, LsqFit, Distributions, DSP

# Functions
extend_range(r,δ) = (first(r) - δ):step(r):(last(r) + δ)

# Import simulation. Parameters

dt = 0.1
Nh::Int = 1e6
theoreticalN0::Bool = 0
theoreticalN0 || (N0::Int = 1e4)
R0 = 4
r = 40
rInt::Int = ceil(r)
H(x) = exp.(-abs.(x)/r) 
Hkernel = H(-5*rInt:5*rInt)
HkernelHalfLength::Int = floor(length(Hkernel)/2)

mutationRate = 0.5
mutationAv = 0
mutationScale = 1
mutationKernel = Normal(mutationAv, mutationScale)

nonLocalMutProb = 0
nonLocalJump = 0
localKernel = Normal(mutationAv, mutationScale)
localKernelType = string(typeof(localKernel))
localDist = localKernelType[1: findfirst('{', localKernelType) - 1] * string(mutationScale)
kernelType = string(typeof(mutationKernel))
mutationKernel != "piecewise" ? dist = kernelType[1: findfirst('{', kernelType) - 1] * string(mutationScale) : dist = mutationKernel * string(nonLocalJump) * "/" * "nonLocalProb" * string(nonLocalMutProb) * "/" * localDist

xmax::Int64 = 3000
tmax = 2000

boundaryConditions = Dict([(1, "bounded"), (2, "reflecting"), (3, "absorbing")])
bc = 3
boundaryCondition = boundaryConditions[bc]

dir = "simulations/1D/" * dist * "/dt$(dt)_N0$(theoreticalN0 ? "theo" : N0)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_xSize$(xmax)"

isdir(dir) || error("The given parameter combination has not been simulated yet (or the path to the directory is incorrect, check pwd)")

dtSampling = 1
idxSampling::Int = dtSampling/dt

fileNxt = "Nxt.csv"
fileNxtSampled = "NxtSampled_dtSampling$(dtSampling).csv"
filexnx = "xnx_dtSampling$(dtSampling).csv"
filehx = "hx_dtSampling$(dtSampling).csv"

tSampled, xtSampled, NtSampled = Vector.(eachcol(CSV.read(joinpath(dir, fileNxtSampled), CSV.Tables.matrix)))
t, xt, Nt = Vector.(eachcol(CSV.read(joinpath(dir, fileNxt), CSV.Tables.matrix)))

nx = CSV.read(joinpath(dir, filexnx), CSV.Tables.matrix)
x = nx[:, 1]
nx = transpose(nx[:, 2:end])
hx = transpose(CSV.read(joinpath(dir, filehx), CSV.Tables.matrix))
rhox = hx ./ sum(hx, dims=2)

nx0 = nx[1, :]
rhox0 = rhox[1, :]
Nt0 = Nt[1]

x = range(xmax - length(nx0) + 1, xmax)

# Plot the results of the simulation. Plot config

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=1, framestyle=:box, label=nothing, grid=false)
gr()

# Figure directory
figDir = "figures/1D/" * dist * "/dt$(dt)_N0$(theoreticalN0 ? "theo" : N0)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_xSize$(xmax)"
isdir(figDir) || mkpath(figDir)

# First plot: summary

tTransient = 100
idxTransient = Int(tTransient / dt) + 1
isAbsorbed = any(isnan.(xt))
if isAbsorbed
    idxAbsorbed = findfirst(isnan.(xt))
    tAbsorbed = min(t[end - 2], t[idxAbsorbed])
    idxAbsorbedSampled::Int = round(idxAbsorbed/(dtSampling/dt)) 
end
maxIdx = (isAbsorbed ?  Int(round((tAbsorbed - tTransient) / dt)) + 1 : length(t) - 2)
fastAbsorption = maxIdx < idxTransient
v = (xt[3:end] .- xt[1:end-2]) ./ 2dt
vAverage = mean(v[idxTransient-1:maxIdx])
vStd = std(v[idxTransient-1:maxIdx], mean = vAverage)
NAverage = mean(Nt[idxTransient:maxIdx])
NStd = std(Nt[idxTransient:maxIdx], mean = NAverage)
hAverage = mean(hx[end,Int.(round.(xt[idxTransient:maxIdx]))])
hStd = std(hx[end,Int.(round.(xt[idxTransient:maxIdx]))], mean = hAverage)

p0 = plot(x, nx0 ./ Nt0, colour=:lightsalmon, title="Virus-immune chasing, " * raw"$r = " * "$(r)," * raw"R_0 = " * "$(R0)" * raw"$" * (!fastAbsorption ? ",\n" * raw"$\bar{v} = " * "$(round(vAverage, sigdigits= 2))" * raw"\pm" * "$(round(vStd, sigdigits= 1))," * raw"\bar{N} = " * "$(Int(round(NAverage, sigdigits=2)))" * raw"\pm" * "$(Int(round(NStd, sigdigits= 1)))" * raw"$" * "\n" * raw"$\bar{h}=" * "$(Int(round(hAverage, sigdigits=2)))" * raw"\pm" * "$(Int(round(hStd, sigdigits= 1)))"  *  raw".$" : raw".") , ylabel=raw"$n(x)/N(t), \rho(x)$", xlabel="x", top_margin=20Plots.px, label=raw"$n(x)/N(t)$ initial condition", legend_position=:topright)
plot!(x, nx[end, :] ./ Nt[end], colour=:coral, label=raw"$n(x)/N(t)$ final distribution")
plot!(x, rhox0, colour=:lightsteelblue, label=raw"$\rho(x)$ initial condition")
plot!(x, rhox[end, :], colour=:steelblue, label=raw"$\rho(x)$ final distribution")

p1 = plot(t, Nt, color=:steelblue4, ylabel=raw"$N(t)$", xlabel=raw"$t$")
hline!(p1, [NAverage], color = :black, ls = :dash)
p2 = plot(t, xt, color=:coral, ylabel=raw"$\bar{x}(t)$", xlabel=raw"$t$")

p3 = plot(x[hx[end, :].!=0], hx[end, :][hx[end, :].!=0], color=:steelblue, ylabel=raw"$h(x)$", xlabel=raw"$x$")
hline!(p3, [hAverage], color = :black, ls = :dash)

p4 = plot(t[2:end-1], v, color = :coral, ylabel = raw"$v(t)$", xlabel = raw"$t$", xlims = (t[1], t[end]), ylims = (minimum(v[1:(fastAbsorption ? end : maxIdx)]), maximum(v[1:(fastAbsorption ? end : maxIdx)])))
hline!(p4, [vAverage], color = :black, ls = :dash)
annotate!()

l = @layout [a{0.5h}
    [grid(2, 2)]]
p = plot(p0, p1, p2, p3, p4, layout=l, size=(600, 600))
display(p)

savefig(p, joinpath(figDir, "ViralImmuneWaveOverview.png"))

# Second plot: wave shape in the final position

isAbsorbed ? idx::Int = round(idxAbsorbedSampled/2) : idx::Int = length(nx[:,1])

any(nx[idx, :] .> 0) ? (initialRange = findfirst(nx[idx, :] .!= 0):findlast(nx[idx, :] .!= 0)) : (initialRange = 1:length(nx[idx, :]))
extendedRange = initialRange # extend_range(initialRange, 5)
Hkernel = H(-5*rInt:5*rInt)
c = conv(hx[idx, :], Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
f =   R0 .* exp.(-c ./ Nh) .- 1

p1 = plot(x[extendedRange], hx[idx, extendedRange] ./ Nh, colour=:steelblue, ylabel=raw"$n(x,t)/N(t),\quad f(t)\mathrm{\,\,(a.u.)}$", xlabel=raw"$x$", label=raw"$n(x)/N(t)$", legend_position=:right, leftmarign = 20Plots.pt, rightmargin = 20Plots.pt, lw=2, frame = :semi)
plot!(x[extendedRange], nx[idx, extendedRange] ./ NtSampled[idx], colour=:orangered, ylabel = raw"Densities", lw=2, label=raw"$h(x)/N_h$")
plot!([], [], label=raw"$f(x)$", colour=:lightsalmon, lw = 2)
plot!(twinx(), x[extendedRange], f[extendedRange] ./ R0, colour=:lightsalmon, lw=2, ylabel = raw"$f(x,t)/R_0$", frame = :semi)
plot!(widen = :false)
hline!([ylims(p1)[2]], lc=:black, lw=1.5)

p = plot(p1, foreground_color_legend = nothing)
display(p)

savefig(p, joinpath(figDir, "WaveShape.png"))

# # Animation of the wave evolution

# anim = @animate for i in 1:(isAbsorbed ? min(idxAbsorbedSampled + 100, size(nx)[1])  : size(nx)[1])
#     p = plot(x, nx[i, :] ./ NtSampled[i], colour=:coral, ylims=[0, 0.2], ylabel=raw"$n(x,t)/N(t), \rho(x,t)$", xlabel=raw"$x$")
#     plot!(x, rhox[i, :], colour = :steelblue, background_color_legend = :white)
#     pTwinx = twinx()
#     plot!(pTwinx, [], [], color = :coral, label = raw"$n(x,t)/N(t)$")
#     plot!(pTwinx, [], [], color = :steelblue, label = raw"$\rho(x,t)$")
#     plot!(pTwinx, xtSampled[1:min(i,findlast(NtSampled .> 0))], NtSampled[1:min(i,findlast(NtSampled .> 0))], color = :grey, label=raw"$N(t)$", legend_position = :topleft, ylims=[1, maximum(NtSampled)], yaxis=:log, ylabel=raw"$N(t)$")
# end

# g = gif(anim, joinpath(figDir, "ViralImmuneWaveAnimation.gif"), fps=30*Int(round(5/dtSampling)))
# display(g)


# Animation of the wave evolution without normalising n(x)

animNonNorm = @animate for i in 1:(isAbsorbed ? min(idxAbsorbedSampled + 100, size(nx)[1])  : size(nx)[1])
    p = plot(x, nx[i, :], colour=:coral, ylims=[0, maximum(nx)], ylabel=raw"Viral density", xlabel=raw"$x$")
    plot!(twinx(), x, hx[i, :] ./ Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"Immune memories", ylims = [0, 1])
    plot!([], [], color = :coral, label = raw"$n(x,t)$")
    plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
end

g = gif(animNonNorm, joinpath(figDir, "ViralImmuneWaveAnimationNoNormalised.gif"), fps = 30)
display(g)