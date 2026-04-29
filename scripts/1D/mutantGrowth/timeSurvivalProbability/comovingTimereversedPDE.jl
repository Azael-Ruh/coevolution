using SpecialFunctions, Plots, LsqFit
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function probabilityUpdateRule!(dmdt, x, t, wVect, params)
    v = params[1]
    D = params[2]
    s = params[3]
    dx = params[4]
    dwdt[1] = s .* x[1] * wVect[1] - wVect[1]^2 + D.*(wVect[2] - wVect[1])./dx^2 - v*(wVect[2] - wVect[1])/dx
    dwdt[end] = s .* x[end] * wVect[end] - wVect[end]^2
    dwdt[2:end-1] .=s .* (x[2:end-1]) .* wVect[2:end-1] .- wVect[2:end-1].^2 .+ D .* (wVect[3:end] .- 2 .* wVect[2:end-1] .+ wVect[1:end-2])./dx^2 .- v .* (wVect[3:end] - wVect[1:end-2]) ./ 2dx
    dwdt
end

v = 0.2
D = 0.1
s = 1e-3
dx = 1
x = -200:dx:250
parameters = (v, D, s, dx)

tmax = 10000
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
    global wLoc = max.(wLoc, 0)

    if ((i+1) % idxSampling == 1)
        idx::Int = (i / idxSampling) + 1
        w[round(Int, t[i+1] ./ dtSampling) + 1, :] = wLoc
    end
    println("i = $i")
end

vs = v*s
wTheoApprox = (r, rf) -> (rf <= 0. ? 20 : vs*exp(r^2/2vs)/(sqrt(pi*vs/2)*(erfi(r/sqrt(2vs)) - erfi((r-2rf)/sqrt(2vs)))))

function funcTofit(r, p)
    rf::Float64 = p[1]
    wf::Vector{Float64} = wTheoApprox.(r, rf)
end

rfArray = collect(v.*tSampling ./2 .* s - D*s^2 .* tSampling.^2/5)
for i in eachindex(tSampling)[2:end]
    fit = LsqFit.curve_fit(funcTofit, s.*x, w[i, :], [rfArray[i]])
    rfArray[i] = fit.param[1]
    println("Fitting timestep $i")
end


# plotConfig()
# animation = @animate for i in eachindex(tSampling)[1:5:end]
#     p = plot(s.*x, wTheoApprox.(s.*x, rfArray[i]), c = :black, label = "Fitted boundary solution")
#     plot!(p, s.*x, w[i,:], colour=:coral, ylabel=raw"$-\log(1-\mathbb{P}_\mathcal{S}(x_0, t))$", xlabel=raw"$sx_0$", label = "Numerical solution")
# end

# g = gif(animation)
# display(g)

# Find the position of the shoulder

# rf = Vector{Float64}(undef, length(tSampling))
# for i in eachindex(rf)
#     if any(w[i,:] .< s.*x./2)
#         idxFirst = findfirst(w[i,:] .- s.*x./2 .<= 0)
#         rf[i] = s*x[idxFirst + findfirst(w[i,idxFirst:end] .- s.*x[idxFirst:end]./2 .>= 0)]
#     end
# end