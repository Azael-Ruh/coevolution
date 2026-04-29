using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function probabilityUpdateRule!(dmdt, x, t, wVect, params)
    D = params[1]
    s = params[2]
    dx = params[3]
    dwdt[1] = s .* x[1] * wVect[1] - wVect[1]^2 + D.*(wVect[2] - wVect[1])./dx^2 
    dwdt[end] = s .* x[end] * wVect[end] - wVect[end]^2
    dwdt[2:end-1] .=s .* (x[2:end-1]) .* wVect[2:end-1] .- wVect[2:end-1].^2 .+ D .* (wVect[3:end] .- 2 .* wVect[2:end-1] .+ wVect[1:end-2])./dx^2 
    dwdt
end

D = 20
s = 1e-3
dx = 1
x = -200:dx:200
parameters = (D, s, dx)

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

Ds = D * s^2
wTheoFit = (r, tau) -> (r == 0 ? 1/tau : r/(1-exp(-r*tau)))
astar = w[end, 201] ./ Ds^(1/3) ./ airyai(0) # -(1/(2airyaiprime(0)))
A = t -> astar*(Ds)^(1/3)/(1 - exp( - astar*(Ds)^(1/3)*t))
wCorrection = r -> (r < 0 ? airyai(-r/Ds^(1/3)) : airyai(r/Ds^(1/3)))
wTheo = (r, tau) -> wTheoFit(r, tau) + (A(tau)- 1/tau)*wCorrection(r)

plotConfig()
wTheoFit = (r, tau) -> (r == 0 ? 1/tau : r/(1-exp(-r*tau)))
astar =  -(1/(2airyaiprime(0)))
A = t -> astar*(Ds)^(1/3)/(1 - exp( - astar*(Ds)^(1/3)*t))
wCorrection = r -> (r < 0 ? airyai(-r/Ds^(1/3)) : airyai(r/Ds^(1/3)))
wTheo = (r, tau) -> wTheoFit(r, tau) + max((A(tau) - 1/(airyai(0)*tau)), 0)*wCorrection(r)

plotConfig()
animation = @animate for i in eachindex(tSampling)
    p = plot(s.*x, w[i,:], colour=:coral, ylabel=raw"$-\log(1-\mathbb{P}_\mathcal{S}(x_0, t))$", xlabel=raw"$sx_0$", label = "Numerical solution")
    plot!(p, s.*x, wTheoFit.(s.*x, tSampling[i]), c = :black, ls = :dash, label = "No mutation solution")
    plot!(p, s.*x, wTheo.(s.*x, tSampling[i]), c = :black, label = "Airy")
end 
g = gif(animation)