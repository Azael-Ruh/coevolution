using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function probabilityUpdateRule!(dmdt, x, t, wVect, params)
    s = params[1]
    dx = params[2]
    dwdt[1] = s .* x[1] * wVect[1] - wVect[1]^2
    dwdt[end] = s .* x[end] * wVect[end] - wVect[end]^2
    dwdt[2:end-1] .= s .* (x[2:end-1]) .* wVect[2:end-1] .- wVect[2:end-1].^2
    dwdt
end

s = 1e-3
dx = 1
x = -500:dx:500
parameters = (s, dx)

tmax = 1000
dt = 0.01
dtSampling = 1
t = 0:dt:tmax
tSampling = 0:dtSampling:tmax
idxSampling = Int(dtSampling ./ dt)

A0 = 20
w0 = A0.*ones(length(x))
w = Array{Float64}(undef, length(tSampling), length(x))
w[1,:] = w0

wLoc::Vector{Float64} = w0
dwdt = zero(wLoc)
for i in 1:length(t)-1
    global wLoc = wLoc + probabilityUpdateRule!(dwdt, x, t[i], wLoc, parameters) .* dt
    global wLoc .= max.(wLoc, 0)

    if ((i+1) % idxSampling == 1)
        idx::Int = (i / idxSampling) + 1
        w[round(Int, t[i+1] ./ dtSampling) + 1, :] = wLoc
    end
    println("i = $i")
end

wTheo = (r, tau) -> r/(1-exp(-r*tau))

plotConfig()
animation = @animate for i in eachindex(tSampling)
    p = plot(s.*x, w[i,:], colour=:coral, ylabel=raw"$-\log(1-\mathbb{P}_\mathcal{S}(x_0, t))$", xlabel=raw"$sx_0$", label = "Numerical solution")
    plot!(p, s.*x, wTheo.(s.*x, tSampling[i]), c = :black, label = "Analytical solution")
end 
g = gif(animation)