using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function mutantUpdateRule!(dmdt, x, t, mVect, params)
    v = params[1]
    D = params[2]
    s = params[3]
    dx = params[4]
    dmdt[1] = 0
    dmdt[end] = 0
    dmdt[2:end-1] .= s .* (x[2:end-1] .- v * t) .* mVect[2:end-1] + D .* (mVect[1:end-2] - 2mVect[2:end-1] + mVect[3:end])./dx^2
    dmdt
end

v = 0.1
D = 0.1
s = 1e-3
dx = 0.5
x = -25:dx:1000

Nh = 1000000
r = 40
H(x) = exp.(-abs.(x)/r)
if r == 0
    Hkernel = [1]
else
    Hkernel = H(-5*ceil(r):5*ceil(r))
end
HkernelHalfLength::Int = floor(length(Hkernel)/2)

tmax = 500
dt = 0.004
dtSampling = 0.1
t = 0:dt:tmax
tSampling = 0:dtSampling:tmax
idxSampling = Int(dtSampling ./ dt)

x0 = 60
M0Gaussian = 1/(x0*s)
gaussianCond(x,x0,var) = M0Gaussian .*sqrt(1/(2 .* pi .* var)) .* exp.(-(x.-x0).^2 ./ var)
deltaCond(x,x0) = 1. .* (x .== x0)
m0 = deltaCond(x,x0)

# m = Array{Float64}(undef, length(tSampling), length(x))
# m[1,:] = m0
M = Vector{Float64}(undef, length(tSampling))
M[1] = sum(m0)
xm = Vector{Float64}(undef, length(tSampling))
xm[1] = sum(x .* m0) ./ M[1]
sigmam = Vector{Float64}(undef, length(tSampling))
sigmam[1] = sum(x.^2 .* m0) ./ M[1] - xm[1].^2
m3 = Vector{Float64}(undef, length(tSampling))
m3[1] = sum((x .- xm[1]).^3 .* m0) ./ M[1]
parameters = (v, D, s, dx)
hm = Vector{Float64}(undef, length(tSampling))
hm[1] = 0
cm = Vector{Float64}(undef, length(tSampling))
cm[1] = 0
fm = Vector{Float64}(undef, length(tSampling))
fm[1] = s.*x0

mLoc::Vector{Float64} = m0
hLoc = zero(mLoc)
dmdt = zero(mLoc)
for i in 1:length(t)-1
    global mLoc = mLoc + mutantUpdateRule!(dmdt, x, t[i], mLoc, parameters) .* dt
    
    global hLoc = hLoc + mLoc .* dt
    c = conv(hLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
    f = (s .* (x .- v .* t[i]) .+ 1) .* exp.(-c ./ Nh) .- 1

    if ((i+1) % idxSampling == 1)
        idx::Int = (i / idxSampling) + 1
        M[idx] = sum(mLoc)
        xm[idx] = sum(x .* mLoc) ./ M[idx]
        sigmam[idx] = sum(x.^2 .* mLoc) ./ M[idx] - xm[idx].^2
        m3[idx] = sum((x .- xm[idx]).^3 .* mLoc) ./ M[idx]

        idxXm = findfirst(x .> xm[idx])
        alpha = (x[idxXm] - xm[idx])/dx
        hm[idx] = hLoc[idxXm - 1]*alpha + hLoc[idxXm]*(1-alpha)
        cm[idx] = c[idxXm - 1]*alpha + c[idxXm]*(1-alpha)
        fm[idx] = f[idxXm - 1]*alpha + f[idxXm]*(1-alpha)
        # m[Int(t[i] ./ dtSampling) + 1, :] = mLoc
    end
    # println("i = $i")
end

plotConfig()
Mp =plot(tSampling, M, yscale = :log10, colour = :coral, lw= 1, ylabel = raw"$M(t)$", xlabel = raw"$t$", label = "Numerical solution")
plot!(Mp, tSampling, M[1].*exp.(s.*(x0.*tSampling .- v.*tSampling.^2 ./ 2 .+ (s*D).*tSampling.^3 ./ 3)), c = :black, label = "Gaussian approximate solution")
xp = plot(tSampling, xm, xlabel = raw"$t$", c = :steelblue, ylabel = raw"$\langle x_m\rangle(t)$", label = "Numerical solution")
plot!(xp, tSampling, x0 .+ (s*D).*tSampling.^2, colour = :black, label = "Gaussian approximate solution")
sigmap = plot(tSampling, sigmam,  xlabel = raw"$t$", c = :steelblue, ylabel = raw"$\sigma_m^2(t)$", label = "Numerical solution")
plot!(sigmap,tSampling, 2D.*tSampling, c = :black, label = "Gaussian approximate solution")
plot!(sigmap, [],[], c = :coral, label = raw"Non-Gaussian correction $sm_3(t)/2D$")
plot!(twinx(), tSampling, s.*m3 ./ 2D, c = :coral, ylabel = "Non-Gaussian correction ratio", label = "", ylims = (0,1))

display(Mp)
display(xp)
display(sigmap)

# Immunity kickoff

MInt = cumsum(M).*dtSampling
plot(tSampling, MInt, yscale = :log10, ylabel = "Mutant lineage size", label = "Numerical lineage size", xlabel = raw"$t$")
plot!(tSampling, M ./ (s .* (xm .- v.*tSampling)), label = "Laplace approximated lineage size") #Laplace approximation
plot!(tSampling, Nh*s.*(xm .- v.*tSampling), label = "Immunity kickoff condition") # Threshold
plot!(twinx(), tSampling, s.*m3 ./ 2D, c = :coral, ylims = (0,1), ylabel = "Non-gaussian correction") # Correction

htheo = M ./ (s .* sigmam)  # immunity at mutant average
cmtheo = [sum(M[1:i] .* exp.(.-(xm[i] .- xm[1:i])./r)) for i in eachindex(tSampling)] .* dtSampling # coverage approximation
cmtheoApprox = M ./ (s .* (xm .- v.*tSampling)) # Laplace approx
th = 1/(s*x0)*log(Nh/1*s^2*(x0-v/(x0*s))^2) # Approximated immunity kickoff

plot(tSampling[2:end], cm[2:end], ylabel = "Immune coverage at average mutant position", xlabel = raw"$t$", label = "Numerical coverage", yscale = :log10, legend_position = :topleft)
plot!(tSampling, cmtheo, label = "Theoretical coverage")
plot!(tSampling, cmtheoApprox, label = "Theoretical coverage approximated")
plot!(tSampling, Nh*s.*(xm .- v.*tSampling), label = "Immunity kickoff condition") # Threshold
vline!([th], c = :black, ls = :dash, lw = 0.5, label = "Theoretical immunity kickoff time")

bulkfm = s .* (xm .- v.*tSampling)
plot(tSampling, fm, ylims = (0, 0.1), ylabel = "Fitness at mutant average position", xlabel = raw"$t$", label = "Numerical fitness")
plot!(tSampling, bulkfm, label = "Bulk-associated fitness")
vline!([th], c = :black, ls = :dash, lw = 0.5, label = "Theoretical immunity kickoff time")