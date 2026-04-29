using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function probabilityUpdateRule!(dmdt, x, t, wVect, params)
    v = params[1]
    D = params[2]
    s = params[3]
    T = params[4]
    dx = params[5]
    dwdt[1] = s .* (x[1] - v * (T - t)) * wVect[1] - wVect[1]^2 + D.*(wVect[2] - wVect[1])./dx^2
    dwdt[end] = s .* (x[end] - v * (T - t)) * wVect[end] - wVect[end]^2
    dwdt[2:end-1] .=s .* (x[2:end-1] .- v * (T - t)) .* wVect[2:end-1] .- wVect[2:end-1].^2 .+ D .* (wVect[3:end] .- 2 .* wVect[2:end-1] .+ wVect[1:end-2])./dx^2
    dwdt
end

v = 0.1
D = 0.1
s = 1e-3
T = 1000
dx = 1
x = -200:dx:200
parameters = (v, D, s, T, dx)

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

    if ((i+1) % idxSampling == 1)
        idx::Int = (i / idxSampling) + 1
        w[round(Int, t[i+1] ./ dtSampling) + 1, :] = wLoc
    end
    println("i = $i")
end

plotConfig()