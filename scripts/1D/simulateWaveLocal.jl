include("../../code/simulate/coevolution1DSimulationTools.jl")
using FFTW

r = 18
R0 = 1.35
mutationRate = 0.17
xmax::Int64 = 220
Nh::Int64 = 1e7

nonLocalJump = 30
nonLocalMutProb = 5e-6
localKernel = Normal(0,1)
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

tmax = 400
dt = 0.1
dtSampling = 1

# (Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxLoc, hxLoc) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

(nx, hx) = simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
Nt = vec(sum(nx, dims = 2))

p = plotSimulationSummary(nx, hx, xmax, r, R0, dtSampling = dtSampling)
display(p)
q = plotWaveShape(nx, hx, xmax, r, R0, Nh)
g = animateSimulationLog(nx, hx, x, Nh)
1

# autop = plot(autocor(Nt, 0:floor(Int,length(Nt)/2)), ylabel = raw"$\langle N(t)N(t+\tau)\rangle$", xlabel = raw"$\tau$", title = "Normalised autocorrelation for R0 = $R0, r = $r")
# display(autop)
# F = fftshift(fft(Nt .- mean(Nt)))
# freqs =  fftshift(fftfreq(length(Nt), 1/dtSampling))
# fourierp = plot(freqs, abs.(F), xlims = (-0,0.02), xlabel = raw"$f$", ylabel = raw"$\mathcal{F}[N(t)- \langle{N}\rangle](f)$", title = "Fourier transform for R0 = $R0, r = $r")
# display(fourierp)

# nRuns = 10
# autoCorNt = autocor(Nt, 0:floor(Int,length(Nt)/2))
# survivedRuns = 1
# autop2 = plot()
# fourierp2 = plot()
# for run in 1:nRuns
#     (NtLoc, xtLoc, _, _, absorbedStateLoc, _, _, _) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)
#     autoCorNtLoc = autocor(NtLoc, 0:floor(Int,length(Nt)/2))
#     global autoCorNt += autoCorNtLoc.*(absorbedStateLoc == 0)
#     FLoc = fftshift(fft(NtLoc .- mean(NtLoc)))
#     global F += FLoc .* (absorbedStateLoc == 0)
#     global survivedRuns += 1 * (absorbedStateLoc == 0)
#     println("Finished run $run")
#     if absorbedStateLoc == 0
#         plot!(autop2, autoCorNtLoc, label = "", c = :gray, lw = 0.5, alpha = 0.5)
#         plot!(fourierp2, freqs, abs.(F), label = "", c = :gray, lw = 0.5, alpha = 0.5)
#     end
# end
# F = F ./ survivedRuns
# autoCorNt = autoCorNt ./ survivedRuns

# plot!(autop2, autoCorNt, ylabel = raw"$\langle N(t)N(t+\tau)\rangle$", xlabel = raw"$\tau$", title = "Normalised autocorrelation for R0 = $R0, r = $r, nRuns = $survivedRuns", c = :black, lw = 1.5)
# display(autop2)
# plot!(fourierp2, freqs, abs.(F), xlims = (-0,0.02), xlabel = raw"$f$", ylabel = raw"$\mathcal{F}[N(t)- \langle{N}\rangle](f)$", title = "Fourier transform for R0 = $R0, r = $r, nRuns = $survivedRuns", c = :black, lw = 1.5)
# display(fourierp2)