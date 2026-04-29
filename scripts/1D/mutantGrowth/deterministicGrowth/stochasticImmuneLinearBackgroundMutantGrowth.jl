using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function simulateStochasticMutant(v, r, s, Nh, mutationRate, mutationKernel, x, tmax, dt, dtSampling, dtSamplingDistribution, x0, M0, initialCond; dx = 1)

    maxIdx = length(x)

    # Mutations
    mutationScale = std(mutationKernel)
    D = mutationRate * mutationScale^2 / 2

    # Immunity
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    t = 0:dt:tmax
    tSampling = 0:dtSampling:tmax
    tSamplingDistribution = 0:dtSamplingDistribution:tmax
    idxSampling = Int(dtSampling ./ dt)
    idxSamplingDistribution = Int(dtSamplingDistribution ./ dt)

    m0 = round.(Int, M0 .* initialCond(x, x0))

    m = Array{Int64}(undef, length(tSamplingDistribution), length(x))
    m[1,:] = m0
    M = Vector{Int64}(undef, length(tSampling))
    M[1] = sum(m0)
    xm = Vector{Float64}(undef, length(tSampling))
    xm[1] = sum(x .* m0) ./ M[1]
    varm = Vector{Float64}(undef, length(tSampling))
    varm[1] = sum(x.^2 .* m0) ./ M[1] - xm[1].^2
    m3 = Vector{Float64}(undef, length(tSampling))
    m3[1] = sum((x .- xm[1]).^3 .* m0) ./ M[1]
    parameters = (v, D, s, dx)
    hm = Vector{Float64}(undef, length(tSampling))
    hm[1] = 0
    cm = Vector{Float64}(undef, length(tSampling))
    cm[1] = 0
    fm = Vector{Float64}(undef, length(tSampling))
    fm[1] = s.*x0

    mLoc::Vector{Int64} = m0
    hLoc = zero(mLoc)
    mutantExtinct = false
    tExtinct = NaN
    for i in 1:length(t)-1

        # Growth and death
        c = conv(hLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
        RBack = 1 .+ s.*(x .- v.*t[i] )
        Reff = RBack.*exp.(-c ./Nh)
        mGrowth = rand.(Poisson.(Reff .* mLoc .* dt))
        mDeath = rand.(Poisson.(mLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        mLoc = max.(mLoc .+ mGrowth .- mDeath, 0)
            
        # Immune evolution
        hLoc = hLoc + mDeath

        mutantExtinct = iszero(mLoc)
        if mutantExtinct
            tExtinct = t[i+1]
            idx = ceil(Int, i / idxSampling) + 1
            M[idx:end] .= 0
            xm[idx:end] .= xm[idx-1]
            varm[idx:end] .= 0
            m3[idx:end] .= 0

            idxXm = findfirst(x .> xm[idx])
            alpha = (x[idxXm] - xm[idx])/dx
            hm[idx:end] .= hLoc[idxXm - 1]*alpha + hLoc[idxXm]*(1-alpha)
            cm[idx:end] .= c[idxXm - 1]*alpha + c[idxXm]*(1-alpha)

            f = Reff .- 1
            fm[idx:end] .= f[idxXm - 1]*alpha + f[idxXm]*(1-alpha)
            
            idx = ceil(Int, i / idxSamplingDistribution) + 1
            m[idx:length(tSamplingDistribution), :] .= 0
            
            println("Mutant extinct")
            break
        end

        # Mutations
        mMutated = sparsevec(rand.(Binomial.(mLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacements = getDisplacement.(iszero(mMutated) ? [(0, 0)] : tuple.(mMutated.nzind, mMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        mJump = displacementToJump.(mutationDisplacements, maxIdx, dx = dx) # 4.643 ms
        mLoc = mLoc - Array(mMutated) + Array(sum(mJump)) # Move mutated viruses

        if ((i+1) % idxSampling == 1)
            # println("$i")
            idx::Int = (i / idxSampling) + 1
            M[idx] = sum(mLoc)
            xm[idx] = sum(x .* mLoc) ./ M[idx]
            varm[idx] = sum(x.^2 .* mLoc) ./ M[idx] - xm[idx].^2
            m3[idx] = sum((x .- xm[idx]).^3 .* mLoc) ./ M[idx]

            idxXm = findfirst(x .> xm[idx])
            alpha = (x[idxXm] - xm[idx])/dx
            hm[idx] = hLoc[idxXm - 1]*alpha + hLoc[idxXm]*(1-alpha)
            cm[idx] = c[idxXm - 1]*alpha + c[idxXm]*(1-alpha)

            f = Reff .- 1
            fm[idx] = f[idxXm - 1]*alpha + f[idxXm]*(1-alpha)
        end
        if ((i+1) % idxSamplingDistribution == 1)
            idx = Int(i / idxSamplingDistribution) + 1
            m[idx, :] = mLoc
        end
    end

    return t, tSampling, M, xm, varm, m3, hm, cm, fm, tSamplingDistribution, m, tExtinct
end

v = 0.1
s = 2e-3
r = 40
Nh = 1000000
dx = 1
x = -40:dx:120

# Mutations
mutationRate = 0.2
mutationKernel = 2*Bernoulli(0.5) - 1
mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2

tmax = 250
dt = 0.001
dtSampling = 0.1
dtSamplingDistribution = 1

x0 = 40
M0 = 1 # ceil(Int, 1/(x0*s))
v0 = 10 # 2*D/(x0*s)
gaussianCond(x,x0,var) = exp.(-(x.-x0).^2 ./ 2var) ./ sum(exp.(-(x.-x0).^2 ./ 2var))
deltaCond(x,x0) = 1 .* (x .== x0)
initialCondition = deltaCond #(x,x0) -> gaussianCond(x, x0, v0)

nRuns = 2000

mGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSamplingDistribution)+1, length(x))
MGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)
xmGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)
varmGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)
m3Grid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)
tExtinction = Vector{Float64}(undef, nRuns)
cmGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)
fmGrid = Array{Float64}(undef, nRuns, floor(Int, tmax/dtSampling)+1)

t = 0:dt:tmax
tSampling = 0:dtSampling:tmax
tSamplingDistribution = 0:dtSamplingDistribution:tmax
for run in 1:nRuns
    println("Run $run started")
    t, tSampling, MGrid[run, :], xmGrid[run,:], varmGrid[run,:], m3Grid[run,:], hm, cmGrid[run,:], fmGrid[run,:], tSamplingDistribution, mGrid[run, :, :], tExtinction[run] = simulateStochasticMutant(v, r, s, Nh, mutationRate, mutationKernel, x, tmax, dt, dtSampling, dtSamplingDistribution, x0, M0, initialCondition, dx = dx)
    println("Run $run finished")
end

# # Plots for multiple run averaging 
plotConfig()

avVarm = sum([varmGrid[i,:] for i in 1:nRuns])./nRuns
avm3 = sum([m3Grid[i,:] for i in 1:nRuns])./nRuns
varCorrected = cumsum(sum([replace(2D.*(1 .- 1 ./ MGrid[run, :]), -Inf=>0) for run in 1:nRuns]) ./ nRuns) .* dtSampling
varCorrected2 = cumsum(sum([replace(2D.*(1 .- 2 ./ MGrid[run, :]), -Inf=>0) for run in 1:nRuns]) ./ nRuns) .* dtSampling
varCorrection = s.*cumsum(sum([replace(m3Grid[run, :] .* (1 .- 1 ./ MGrid[run,:]), NaN=>0) for run in 1:nRuns]) ./ nRuns) .* dtSampling
varCorrection2 = cumsum(sum([replace(-2 .* varmGrid[run, :] ./ MGrid[run, :], NaN=>0) for run in 1:nRuns]) ./ nRuns) .* dtSampling
avVarmSurvived = sum([varmGrid[i,:].*isnan(tExtinction[i]) for i in 1:nRuns])./sum(isnan.(tExtinction))
varCorrectedSurvived = cumsum(sum([isnan(tExtinction[run]) .* replace(2D.*(1 .- 1 ./ MGrid[run, :]), -Inf=>0) for run in 1:nRuns]) ./ sum(isnan.(tExtinction))) .* dtSampling
varCorrectedSurvived2 = cumsum(sum([isnan(tExtinction[run]) .* replace(2D.*(1 .- 1 ./ MGrid[run, :]), -Inf=>0) for run in 1:nRuns]) ./ sum(isnan.(tExtinction))) .* dtSampling
varCorrectionSurvived = s.*cumsum(sum([isnan(tExtinction[run]) * replace(m3Grid[run, :] .* (1 .- 1 ./ MGrid[run,:]), NaN=>0) for run in 1:nRuns]) ./ sum(isnan.(tExtinction))) .* dtSampling
varCorrectionSurvived2 = cumsum(sum([isnan(tExtinction[run]) .* replace(-2 .* varmGrid[run, :] ./ MGrid[run, :], NaN=>0) for run in 1:nRuns]) ./ sum(isnan.(tExtinction))) .* dtSampling
varp = plot(tSampling, varmGrid',  xlabel = raw"$t$", c = :gray, ylabel = raw"$\sigma_m^2(t)$", label = permutedims(["" for i in 1:nRuns]), alpha = 0.4)
plot!(varp, tSampling, avVarm, colour = :black, label = "Simulation average", lw = 2)
plot!(varp, tSampling, avVarm[1] .+ 2D.*tSampling, c = :coral, label = "Gaussian approximate solution", lw = 2)
# plot!(varp, tSampling, varCorrected .- varCorrection, c = :steelblue, lw = 2, label = "Integrated equaton with stochastic and survival correction 1")
plot!(varp, tSampling, varCorrected2 .+ varCorrection2, c = :turquoise, lw = 2, label = "Integrated equaton with stochastic and survival correction")
plot!(tSampling, avVarmSurvived, c = :springgreen3, lw = 2, label = "Simulation survived")
plot!(tSampling, varCorrectedSurvived2 .+ varCorrectionSurvived2, c = :violetred1, lw = 2, label = "Integrated equation for survived runs")

# Position plots
avPos = sum([xmGrid[i,:] for i in 1:nRuns])./nRuns
xmTheo = x0 .+ s.*D.*tSampling.^2
xmCorrection = -cumsum( s.* sum([replace(varmGrid[run, :] ./ MGrid[run,:], NaN=> 0) for run in 1:nRuns] ./ nRuns)) .* dtSampling
avPosSurvived = sum([xmGrid[i,:].*isnan(tExtinction[i]) for i in 1:nRuns])./sum(isnan.(tExtinction))
xp = plot(tSampling, xmGrid',  xlabel = raw"$t$", c = :gray, ylabel = raw"$\bar{x}_m(t)$", label = permutedims(["" for i in 1:nRuns]), alpha = 0.4)
plot!(xp, tSampling, avPos, colour = :black, label = "Simulation average", lw = 2)
plot!(xp, tSampling, x0 .+ s.*D.*tSampling.^2, c = :coral, label = "Gaussian approximate solution", lw = 2)
plot!(xp, tSampling, x0 .+ s.*cumsum(avVarm).*dtSampling, c = :steelblue, label = raw"Integrated average variance", lw = 2)
plot!(xp, tSampling, avPosSurvived, c = :springgreen3, lw = 2, label = "Simulation survived")
plot!(xp, tSampling, x0 .+ s.*cumsum(sum([varmGrid[run, :] .* isnan(tExtinction[run]) for run in 1:nRuns])) ./ sum(isnan.(tExtinction)) .*dtSampling, c = :turquoise, label = raw"Integrated average variance survived", lw = 2, legend_position = :bottomright)
plot!(xp, ylims = (x0-5,x0+5))

# Size plots
avM = sum([MGrid[i,:] for i in 1:nRuns])./nRuns
Mtheo = M0 .* exp.(s.*(x0.*tSampling .- v.*tSampling.^2 ./ 2 .+ s.*D.*tSampling.^3 ./ 3))
Mintegrated = M0 .+ cumsum(s.* (sum([xmGrid[run, :] .* MGrid[run, :] for run in 1:nRuns])./nRuns .- v.*tSampling.*avM)) .* dtSampling
MimmuneCorrection = - (1/2Nh) .* sum([cumsum(MGrid[run,:]).^2 .* dtSampling.^2 for run in 1:nRuns]) ./ nRuns
MintegratedCorrected = Mintegrated + MimmuneCorrection
MintegratedSim = M0 .+ cumsum(sum([s.*(xmGrid[run, :] .- v.*tSampling).* MGrid[run, :] .- cmGrid[run,:].*MGrid[run,:]./Nh for run in 1:nRuns] ./ nRuns)) .* dtSampling
Mp = plot(tSampling, replace(MGrid, 0=>NaN)',  xlabel = raw"$t$", c = :gray, ylabel = raw"$M(t)$", yscale = :log10, label = permutedims(["" for i in 1:nRuns]), alpha = 0.4)
plot!(Mp, tSampling[avM .> 1], avM[avM .> 1], c = :black, lw = 2, label = "Simulation average")
plot!(Mp, tSampling[MintegratedCorrected .> 1], MintegratedCorrected[MintegratedCorrected .> 1], c = :steelblue, lw = 2, label = raw"Integrated equation at first order with $c(x,t)\approx\int M(t')$")
plot!(Mp, tSampling[MintegratedSim .> 1], MintegratedSim[MintegratedSim .> 1], c = :turquoise, lw = 2, label = "Integrated equation at first order with simulation coverage")
plot!(Mp, tSampling, Mtheo, c = :coral, lw = 2, label = "Gaussian theory")

# Average shape \equiv survival shape
avm = sum([mGrid[i,:,:] for i in 1:nRuns])./nRuns
avmtheo = Mtheo[end] .* exp.(-(x .- x0 .- s*D*tmax^2).^2 ./ (4*D*tmax)) ./ sqrt(4*pi*D*tmax) .* dx
avmp = plot(x, avm[1:5:end,:]', label = "", xlabel = raw"$x$", ylabel = raw"$\langle m(x,t)\rangle$", palette = palette(:viridis, length(tSamplingDistribution[1:5:end])))
plot!(avmp, x, avm[end,:], label = "Last average profile", c = :black, lw = 2)
plot!(avmp, x, avmtheo, label = "Gaussian theoretical profile", c = :coral, lw = 2)

avmSurvived = avmSurvived = sum([isnan(tExtinction[i]) .* mGrid[i,:,:] for i in 1:nRuns])./sum(isnan.(tExtinction))
avmSurvivedp = plot(x, avmSurvived[1:5:end,:]', label = "", xlabel = raw"$x$", ylabel = raw"$\langle m(x,t)| \mathrm{Survival}\rangle$", palette = palette(:viridis, length(tSamplingDistribution[1:5:end])))
plot!(avmSurvivedp, x, avmSurvived[end,:], label = "Last average profile", c = :black, lw = 2)

# Coverage plot
cp = plot(tSampling, replace(cmGrid,0=>NaN)',  xlabel = raw"$t$", c = :gray, ylabel = raw"$c(\bar{x}_m(t),t)$", label = permutedims(["" for i in 1:nRuns]), alpha = 0.4, yscale = :log10)
plot!(tSampling, s.*(x0 .- v.*tSampling).*Nh, ls = :dash, c = :black, lw = 1, label = "Immunity kickoff")
plot!(tSampling[2:end], sum([cmGrid[run,:] for run in 1:nRuns])[2:end]./nRuns, c = :black, lw =2, label = "Simulation average")
plot!(tSampling, cumsum(avM).*dtSampling, c = :turquoise, label = "Theoretical coverage", lw = 2)
plot!(tSampling, avM ./ (s.*(x0 .- v.*tSampling)), c = :steelblue, lw = 2, label = "Laplace-approximated coverage")
plot!(tSampling, Mtheo ./ (s.*(x0 .- v.*tSampling)), c = :coral, lw = 2, label = "Gaussian theory")
plot!(legend_position = :bottomright)

# # Immunity kickoff

# MInt = cumsum(M).*dtSampling
# plot(tSampling, MInt, yscale = :log10, ylabel = "Mutant lineage size", label = "Numerical lineage size", xlabel = raw"$t$")
# plot!(tSampling, M ./ (s .* (xm .- v.*tSampling)), label = "Laplace approximated lineage size") #Laplace approximation
# plot!(tSampling, Nh*s.*(xm .- v.*tSampling), label = "Immunity kickoff condition") # Threshold
# plot!(twinx(), tSampling, s.*m3 ./ 2D, c = :coral, ylims = (0,1), ylabel = "Non-gaussian correction") # Correction

# htheo = M ./ (s .* sigmam)  # immunity at mutant average
# cmtheo = [sum(M[1:i] .* exp.(.-(xm[i] .- xm[1:i])./r)) for i in eachindex(tSampling)] .* dtSampling # coverage approximation
# cmtheoApprox = M ./ (s .* (xm .- v.*tSampling)) # Laplace approx
# th = 1/(s*x0)*log(Nh/1*s^2*(x0-v/(x0*s))^2) # Approximated immunity kickoff

# plot(tSampling[2:end], cm[2:end], ylabel = "Immune coverage at average mutant position", xlabel = raw"$t$", label = "Numerical coverage", yscale = :log10, legend_position = :topleft)
# plot!(tSampling, cmtheo, label = "Theoretical coverage")
# plot!(tSampling, cmtheoApprox, label = "Theoretical coverage approximated")
# plot!(tSampling, Nh*s.*(xm .- v.*tSampling), label = "Immunity kickoff condition") # Threshold
# vline!([th], c = :black, ls = :dash, lw = 0.5, label = "Theoretical immunity kickoff time")

# bulkfm = s .* (xm .- v.*tSampling)
# plot(tSampling, fm, ylims = (0, 0.1), ylabel = "Fitness at mutant average position", xlabel = raw"$t$", label = "Numerical fitness")
# plot!(tSampling, bulkfm, label = "Bulk-associated fitness")
# vline!([th], c = :black, ls = :dash, lw = 0.5, label = "Theoretical immunity kickoff time")


## Animation

# animation = @animate for i in 1:length(tSamplingDistribution)
#     p = plot(x, m[i, :], colour=:salmon, ylims=[0, maximum(m)], ylabel=raw"Mutant density", xlabel=raw"$x$", label = "")
#     # plot!(twinx(), x, hx[i, :] ./ Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"Immune memories", ylims = [0, 1], label = "")
#     plot!([], [], color = :coral, label = raw"$m(x,t)$")
#     # plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
# end