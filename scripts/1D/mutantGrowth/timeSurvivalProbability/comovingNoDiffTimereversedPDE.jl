using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function probabilityUpdateRule!(dmdt, x, t, wVect, params)
    v = params[1]
    s = params[2]
    dx = params[3]
    dwdt[1] = s .* x[1] * wVect[1] - wVect[1]^2 - v*(wVect[2] - wVect[1])/dx
    dwdt[end] = s .* x[end] * wVect[end] - wVect[end]^2
    dwdt[2:end-1] .=s .* (x[2:end-1]) .* wVect[2:end-1] .- wVect[2:end-1].^2 .- v .* (wVect[3:end] - wVect[1:end-2]) ./ 2dx
    dwdt
end

v = 0.4
s = 1e-3
dx = 1
x = -200:dx:200
parameters = (v, s, dx)
tmax = 1000
dt = 0.01
dtSampling = 1
t = 0:dt:tmax
tSampling = 0:dtSampling:tmax
idxSampling = Int(dtSampling ./ dt)

A0 = 20
w0 = A0.*ones(length(x))
wNoDif = Array{Float64}(undef, length(tSampling), length(x))
wNoDif[1,:] = w0

wLoc::Vector{Float64} = w0
dwdt = zero(wLoc)
for i in 1:length(t)-1
    global wLoc = wLoc + probabilityUpdateRule!(dwdt, x, t[i], wLoc, parameters) .* dt
    global wLoc .= max.(wLoc, 0)

    if ((i+1) % idxSampling == 1)
        idx::Int = (i / idxSampling) + 1
        wNoDif[round(Int, t[i+1] ./ dtSampling) + 1, :] = wLoc
    end
    println("i = $i")
end

vs = v*s
wTheoFit = (r, tau) -> r/(1-exp(-r*tau))
wTheo = (r, tau) -> (tau == 0 ? Inf : vs*exp(r^2/2vs)/sum([exp(a^2/2vs) for a in (r-vs*tau):1e-5:r] .* 1e-5))

# plotConfig()
# animation = @animate for i in eachindex(tSampling[1:2:end])
#     p = plot(s.*x, wNoDif[i,:], colour=:coral, ylabel=raw"$-\log(1-\mathbb{P}_\mathcal{S}(x_0, t))$", xlabel=raw"$sx_0$", label = "Numerical solution")
#     plot!(p, s.*x, wTheo.(s.*x, tSampling[i]), c = :black, label = "Analytical solution")
#     plot!(p, s.*x, wTheoFit.(s.*x, tSampling[i]), c = :black, ls = :dash, label = "No bulk solution")
# end 
# g = gif(animation)