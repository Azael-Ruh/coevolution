# ==============================================================
#   Simulate1DWave.jl
# ==============================================================
#
# Project: punctuatedCoevolution (M2 Internship - PhD)
# Author: Max Zayas Orihuela (maxzyas@gmail.com)
# Supervisors: Aleksandra Walczak, Thierry Mora
# Last update: 19/05/2025
# Purpose: produce a 1D simulation of the evolution of an antigenic wave for a rapidly mutating virus under the pressure of the immune system within a population of hosts. The objective of the simulation is to study the effect of non-local mutations.


# ===================================================
#   Imports
# ===================================================
using Distributions, Plots, StatsBase, Interpolations, Tables, CSV, SparseArrays, DSP, SpecialFunctions, NLsolve
import StatsBase: std, params
import Base: broadcastable

# ====================================================================
#                   New types
# ====================================================================

struct picewiseKernel
    type::String
    nonLocalMutProb::Real
    nonLocalJump::Int
    localKernel::Distribution{Univariate, Continuous}
end

function broadcastable(mutKern::picewiseKernel)
    return Ref(mutKern)
end

# function length(kern::picewiseKernel)
#     return 1
# end

# function iterate(kern::picewiseKernel)
#     return (picewiseKernel, nothing)
# end

# function iterate(kern::picewiseKernel, n::Nothing)
#     return n
# end

function params(mutKernel::picewiseKernel)
    (mutKernel.nonLocalMutProb, mutKernel.nonLocalJump, params(mutKernel.localKernel))
end

function std(mutKernel::picewiseKernel)
    return std(mutKernel.localKernel)*(1-mutKernel.nonLocalMutProb)
end
# ====================================================================
#                   Useful functions
# ====================================================================

# General case
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Distribution{Univariate, Continuous})
    nMutated[1] == 0 && return (0,[0])
    (nMutated[1], Distributions.rand!(mutKernel, zeros(nMutated[2])))
end

# Exponential case
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Exponential{Float64})
    nMutated[1] == 0 && return (0,[0])
    mutSign = (Distributions.rand!(Binomial(1, 0.5), zeros(nMutated[2])) .- 0.5) .* 2
    (nMutated[1], Distributions.rand!(mutKernel, zeros(nMutated[2])) .* mutSign)
end

function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Distribution{Univariate, Continuous}, longJumpProb, longJumpLength, localKernel)
    getDisplacement(nMutated, mutKernel)
end

function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::picewiseKernel)
    getDisplacement(nMutated, kernType(mutKernel), mutKernel.nonLocalMutProb, mutKernel.nonLocalJump, mutKernel.localKernel)
end

# Special distributions
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::String, longJumpProb = 0., longJumpLength = 0, localKernel = Normal(0, 1))
    nMutated[1] == 0 && return (0,[0])
    if mutKernel == "piecewise" 
        nNonLocal = rand(Binomial(nMutated[2], longJumpProb))
        return (nMutated[1], [Distributions.rand!(localKernel, zeros(nMutated[2] - nNonLocal)); longJumpLength .* ones(nNonLocal)])
    end
    error("Distribution not yet implemented or mispelled")
end

function getDisplacementPiecewise(nMutated::Tuple{Int64, Int64}, longJumpProb::Float64, longJumpLength::Int64, localKernel::Distribution{Univariate, Continuous})
    xi = rand(nMutated[2])
    nNonLocal = sum(xi .< longJumpProb)
    return (nMutated[1], [Distributions.rand!(localKernel, zeros(nMutated[2] - nNonLocal)); longJumpLength .* ones(nNonLocal)])
end

function displacementToJump(mutDisplacement, maxIdx)
    jumpToIdx::Vector{Int64} = round.(mutDisplacement[2] .+ mutDisplacement[1])
    sparsevec([min(max(jump, 1), maxIdx) for jump in jumpToIdx], [jump > 1 && jump < maxIdx ? 1 : 0 for jump in jumpToIdx], maxIdx)
end

function immuneDeaths(nh, totalDeaths)
    hDeath = Distributions.rand!(DiscreteUniform(1, sum(nh)), zeros(totalDeaths))
    supressIdx::Vector{Int64} = [findfirst(cumsum(nh) .>= selected) for selected in hDeath]
    Array(sparsevec(supressIdx, ones(totalDeaths), length(nh)))
end

function initialDistribution(distType::String, sigma::Real, x, N0, r, R0, Nh, D; Nseed = 1)
    gaussianCond(x) = exp(-x^2/(2*sigma^2))/sqrt(2pi*sigma^2)

    virusIC = Vector{Int64}(undef, length(x))
    immuneIC = Vector{Int64}(undef, length(x))

    if r == 0
        v0 = 2*sqrt(D*(R0-1))
        H0 = N0/v0
    else
        H0 = Nh/r*log(R0)
    end

    if distType == "gaussianBarrier"
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = zero(virusIC)
        immuneIC[findfirst(virusIC .> 0)] = Nh
    elseif distType == "steadyState"
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = round.(H0 .* (1 .+ erf.(.-x ./ sqrt(2*sigma^2))) ./ 2)
    elseif distType == "seedNaiveBackground"
        virusIC = round.(Nseed .* (x .== 0))
        immuneIC = zero(virusIC)
    else
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = zero(virusIC)
    end

    return virusIC, immuneIC
end

function vNEquation(vN, Nh, s, D)
    return [vN[2] - Nh*s*vN[1], vN[1] - D^(2/3)*s^(1/3)*(24max(log(max(vN[2]*(D*s^2)^(1/3),0)),0))^(1/3)]
end

function extinctionFlag(nx, x)
    flagCode = 0
    xmax = maximum(x)

    iszero(nx[end,:]) || return (flagCode, -1)
    
    idxAbsorbed = findfirst([iszero(nx[i,:]) for i in eachindex(nx[:,1])])
    sigma = std(x, FrequencyWeights(nx[idxAbsorbed-1,:]), corrected = true)
    
    escaped = x[findlast(dropdims(sum(nx[1:idxAbsorbed-1,:], dims = 1), dims = 1) .> 0)] > xmax - 3*sigma # If the last values are too close to the boundary the virus probably was absorbed by the boundary = escaped
    if escaped
        flagCode = 2
        println("WARNING: virus escaped") 
    else 
        flagCode = 1
        println("WARNING: virus extinguished")
    end

    return (flagCode, idxAbsorbed)
end

function getInitialCondition(distType::String, R0, r, mutationRate, mutationKernel, Nh, xmax)
    D = mutationRate .* std(mutationKernel)^2/2
    (N0, v, sigma) = getSteadyStateEstimate(R0, r, mutationRate, mutationKernel, Nh)
    x = -Int(max(round(5*r), round(5*sigma))):xmax
    (nx0, hx0) =  initialDistribution(distType::String, sigma::Real, x, N0, r, R0, Nh, D)
    return (nx0, hx0, x)
end

function getSteadyStateEstimate(R0, r, mutRate, mutKernel, Nh)
    mutationScale = std(mutKernel)
    
    D = mutRate*mutationScale^2/2      # Diffusion coefficient
    if r == 0
        v0 = 2*sqrt(D*(R0-1))
        N0 = Nh/100
        sigma = 1
        return (N0, v0, sigma)
    else
        s = log(R0)/r                           # Fitness gradient
    end


    # First calculation assuming linear fitness
    v0 = D^(2/3)*s^(1/3)*(24max(log(Nh/100*(D*s^2)^(1/3)),0))^(1/3)
    if v0 == 0
        v0 = 2*sqrt(D*(R0-1))
    end
    N0 = round(Nh * v0 * s)
    if N0 < 1000
        sigma = sqrt(v0 / s)
        return (N0, v0, sigma)
    end
    
    vNFunction = vN -> vNEquation(vN, Nh, s, D)
    (v0, N0) = nlsolve(vNFunction, [v0, N0]).zero
    if v0 == 0
        v0 = 2*sqrt(D*(R0-1))
        N0 = round(Nh * v0 * s)
        sigma = sqrt(v0 / s)
        return (N0, v0, sigma)
    end

    sigma = sqrt(v0 / s)

    if r > sigma 
        return (max(N0,50), max(v0,10^-3), sigma)
    else
        v0 = 2*sqrt(D*(R0-1))
        N0 = v0*Nh*s
        return (max(N0,50), max(v0,10^-3), sigma)
    end

end

function kernType(mutKernel::Distribution{Univariate, Continuous})
    kernelType = string(typeof(mutKernel))
    return kernelType[1: findfirst('{', kernelType) - 1]
end

function kernType(mutKernel::picewiseKernel)
    return mutKernel.type
end

function getDist(mutKernel::Distribution{Univariate, Continuous})
    return kernType(mutKernel) * "$(round(std(mutKernel)))"
end

function getDist(mutKernel::picewiseKernel)
    return kernType(mutKernel) * mutKernel.nonLocalJump * "prob" * mutKernel.nonLocalMutProb * "/" * getDist(mutKernel.localKernel)
end

function plotConfig()
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
            linewidth=1, framestyle=:box, label=nothing, grid=false)
    gr()
end

# ====================================================================
#                   Main functions
# ====================================================================

function simulateWave(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    # Variable initialisation
    maxIdx = length(x)
    nx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nx[1, :] = nx0
    hx[1, :] = hx0

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Instantaneous fields
    hxLoc = hx0
    nxLoc = nx0

    for i in 2:length(t)

        # Virus growth
        c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
        R = R0 .* exp.(-c ./ Nh)
        nxGrowth = rand.(Poisson.(R .* nxLoc .* dt))
        nxDeath = rand.(Poisson.(nxLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        nxLoc = max.(nxLoc .+ nxGrowth .- nxDeath, 0)

        # Mutations
        nxMutated = sparsevec(rand.(Binomial.(nxLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacements = getDisplacement.(iszero(nxMutated) ? [(0, 0)] : tuple.(nxMutated.nzind, nxMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxJump = displacementToJump.(mutationDisplacements, maxIdx) # 4.643 ms
        nxLoc = nxLoc - Array(nxMutated) + Array(sum(nxJump)) # Move mutated viruses
        
        # Immune evolution
        hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity

        # Sampling
        if i % idxSampling == 1
            nx[Int((i-1)/idxSampling + 1), :] = nxLoc
            hx[Int((i-1)/idxSampling + 1), :] = hxLoc
        end
    end 

    println("Simulation end")
    return (nx, hx)
end

function simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    # Variable initialisation
    maxtIdx = Int(round(tmax/dtSampling+1))
    Nt = Vector{Int64}(undef, maxtIdx)
    xt = Vector{Float64}(undef, maxtIdx)
    sigmat = Vector{Float64}(undef, maxtIdx)
    uTt = x[1] .* zeros(maxtIdx)
    Nt[1] = sum(nx0)
    xt[1] = sum(x .* nx0) ./ Nt[1]
    sigmat[1] = sqrt(sum(x .^2 .* nx0) / Nt[1] - xt[1]^2)
    uTt[1] = x[findlast(nx0 .> 0)] - xt[1] 
    
    maxIdx = length(x)
    absorbedState::Int = 0
    idxAbsorbed::Int = 0

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Instantaneous fields
    hxLoc = hx0
    nxLoc = nx0

    for i in 2:length(t)
        # Virus growth
        c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
        R = R0 .* exp.(-c ./ Nh)
        nxGrowth = rand.(Poisson.(R .* nxLoc .* dt))
        nxDeath = rand.(Poisson.(nxLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        nxLoc = max.(nxLoc .+ nxGrowth .- nxDeath, 0)

        # Mutations
        nxMutated = sparsevec(rand.(Binomial.(nxLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacements = getDisplacement.(iszero(nxMutated) ? [(0, 0)] : tuple.(nxMutated.nzind, nxMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxJump = displacementToJump.(mutationDisplacements, maxIdx) # 4.643 ms
        nxLoc = nxLoc - Array(nxMutated) + Array(sum(nxJump)) # Move mutated viruses
        
        # Immune evolution
        hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity

        # Absorbtion index
        if iszero(nxLoc)
            idxAbsorbed > 0 || (idxAbsorbed = i)
            if uTt[findlast(uTt .> 0)] < sigmat[findlast(uTt .> 0)]
                absorbedState = 2
                println("WARNING: virus escaped")
            else
                absorbedState = 1
                println("WARNING: virus absorbed")
            end
        end

        # Sampling
        if i % idxSampling == 1
            Nt[Int((i - 1) / idxSampling + 1)] = sum(nxLoc)
            xt[Int((i - 1) / idxSampling + 1)] = sum(x .* nxLoc) ./ Nt[Int((i - 1) / idxSampling + 1)]
            sigmat[Int((i - 1) / idxSampling + 1)] = sqrt(sum(x .^2 .* nxLoc) / Nt[Int((i - 1) / idxSampling + 1)] - xt[Int((i - 1) / idxSampling + 1)]^2)
            uTt[Int((i - 1) / idxSampling + 1)] = x[findlast(nxLoc .> 0)] - xt[Int((i - 1) / idxSampling + 1)]
        end
    end 

    println("Simulation end")
    return Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxLoc, hxLoc
end

function simulateWaveStatisticsFull(R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, tTransient; xmax = 0, s = 0, D = 0, initialCond = "steadyState")
    
    s > 0 || (s = log(R0)/r)
    D > 0 || (D = mutationRate*std(mutationKernel)^2/2)

    if xmax == 0
        vFKPP = 2 * sqrt((R0 - 1) * D)
        xmax = 2*max(500, round(Int, vFKPP*tmax + vFKPP^2/(D*s)))
    end

    t = 0:dtSampling:tmax
    (nx0, hx0, x) = getInitialCondition(initialCond, R0, r, mutationRate, mutationKernel, Nh, xmax)
    (Nt, xt, sigmat, uTt, absorbedState, idxAbsorbed, nxBack0, hxBack0) = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

    idxTransient = findfirst(t .>= tTransient)
    if absorbedState == 0
        idxEnd = length(t)
    elseif idxAbsorbed > 2*idxTransient
        idxEnd = length(t) - idxTransient
    else
        print("WARNING: fast absorption")
        idxEnd = idxTransient
    end
    uTAv = mean(uTt[idxTransient:idxEnd])
    NAv = mean(Nt[idxTransient:idxEnd])
    Nstd = std(Nt[idxTransient:idxEnd])
    sigmaAv = mean(sigmat[idxTransient:idxEnd])
    vAv = (xt[idxEnd] - xt[idxTransient]) / (t[idxEnd] - t[idxTransient])

    return (NAv, Nstd, vAv, sigmaAv, uTAv)
end

function saveSimulation(nx, hx, r, R0, mutationRate, mutationKernel, tmax, dt, xmax, initialisation::String; baseFolder = "", fileAppend = "")
    
    x = xmax-size(nx)[2]+1:xmax
    Nt = vec(sum(nx, dims = 2))
    xt = vec(sum(x' .* nx, dims = 2)) ./ Nt
    t = 0:dtSampling:dtSampling.*(length(Nt)-1)

    NxtTable = Tables.table([t xt Nt], header = ["t", "xt", "Nt"])
    xnxTable = Tables.table([x transpose(nx)], header = ["x"; ["t = $(tau)" for tau in t]])
    hxTable = Tables.table(transpose(hx), header = ["t = $(tau)" for tau in t])

    dist = kernType(mutationKernel) * "$(std(mutationKernel))"

    dir = baseFolder * "simulations/1D/" * dist * "/dt$(dt)_dtSamp$(dtSampling)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_" * initialisation
    isdir(dir) || mkpath(dir)

    fileNxt = "Nxt" * fileAppend * ".csv"
    filexnx = "xnx" * fileAppend * ".csv"
    filehx = "hx" * fileAppend * ".csv"

    CSV.write(joinpath(dir, fileNxt), NxtTable)
    CSV.write(joinpath(dir, filexnx), xnxTable)
    CSV.write(joinpath(dir, filehx), hxTable)

    println("Saved files in folder $dir")
end

# ====================================================================
#                   Data analysis tools
# ====================================================================

function producePhaseMatrixes(R0Vect, rVect, nRuns, Nh, mutationRate, mutationKernel, tmax, dt, dtSampling, initialisation; tTransient = 100, bFolder = "")

    survivalProb = zeros((length(R0Vect), length(rVect)))
    vAverage = zeros((length(R0Vect), length(rVect)))
    vStd = zeros((length(R0Vect), length(rVect)))
    NAverage = zeros((length(R0Vect), length(rVect)))
    NStd = zeros((length(R0Vect), length(rVect)))

    for i in eachindex(R0Vect), j in eachindex(rVect), run in 1:nRuns
        
        idxTransient = Int(tTransient / dt) + 1
        t, xt, Nt = loadSimulationNxtData(R0Vect[i], rVect[j], Nh, mutationRate, mutationKernel, tmax, dt, dtSampling, initialisation, baseFolder = bFolder, fileAppend = "_run$(run)")
        
        idxTransient = Int(tTransient / dtSampling) + 1
        isAbsorbed = isnan(last(xt))   # CHECK THIS, IT IS NOT CORRECT!
        if isAbsorbed
            idxAbsorbed = findfirst(isnan.(xt))
            tAbsorbed = t[idxAbsorbed]
            maxIdx = Int(round((tAbsorbed - tTransient) / dtSampling)) + 1
            fastAbsorption = maxIdx < idxTransient
        else
            fastAbsorption = false
            maxIdx = length(t) - 2
        end
        

        if !isAbsorbed
            
            survivalProb[i,j] += 1
            
            v = (xt[3:end] .- xt[1:end-2]) ./ 2dt
            if !fastAbsorption
                vAvRun = mean(v[idxTransient:maxIdx])
                vStdRun = std(v[idxTransient:maxIdx], mean = vAvRun)
                NAvRun = mean(Nt[idxTransient:maxIdx])
                NStdRun = std(Nt[idxTransient:maxIdx], mean = NAvRun)
            else
                vAvRun = xmax / t[idxAbsorbed]
                vStdRun = std(v[1:idxAbsorbed], mean = vAvRun)
                NAvRun = mean(Nt[1:idxAbsorbed])
                NStdRun = std(Nt[1:idxAbsorbed], mean = NAvRun)
            end
            
            vAverage[i,j] += vAvRun
            NAverage[i,j] += NAvRun
            NStd[i,j] += NStdRun
        else
            
        end
    end
        
    vAverage = vAverage ./ survivalProb
    NAverage = NAverage ./ survivalProb
    NStd = NStd ./ survivalProb
    survivalProb = survivalProb./10

    return survivalProb, vAverage, vStd, NAverage, NStd
end

function plotPhaseDiagrams(R0Vect, rVect, survivalProb, vAverage, vStd, NAverage, NStd, mutationKernel; baseFolder = "")

    dist = kernType(mutationKernel) * "$(std(mutationKernel))"
    figDir = baseFolder * "figures/1D/" * dist 

    pSP = heatmap(rVect, R0Vect, 1 .- survivalProb, c = cgrad(:blues, rev=true), xlabel = raw"$r$", ylabel = raw"$R_0$", title = "Extinction probability", titlefontszie = 20)

    pV= heatmap(rVect, R0Vect, vAverage, c = cgrad(:magma), xlabel = raw"$r$", ylabel = raw"$R_0$", title = raw"$\bar{v}$", titlefontszie = 20)

    pN= heatmap(rVect, R0Vect, NAverage, c = cgrad(:magma), xlabel = raw"$r$", ylabel = raw"$R_0$", title = raw"$\bar{N}$", rightmargin = 10Plots.pt, titlefontszie = 20, colorbar_scale = :log10)

    pDeltaN= heatmap(rVect, R0Vect, NStd ./ NAverage, c = cgrad(:magma), xlabel = raw"$r$", ylabel = raw"$R_0$", title = raw"$\Delta N / \bar{N}$", rightmargin = 10Plots.pt, titlefontszie = 20)

    p = plot(pSP, pV, pDeltaN, pN, layout = (2,2), size = (1200, 800), legendfontsize=12, ylabelfontsize = 16, xlabelfontsize = 16, tickfontsize = 12, titlefontszie = 20, dpi = 1000, topmargin = 10Plots.pt, leftmargin = 10Plots.pt)
    
    isdir(figDir) || mkpath(figDir)
    savefig(p, joinpath(figDir, "fullPhaseDiagram.png"))
    return p
end

# ====================================================================
#                   Access data and plotting functions
# ====================================================================


function loadSimulationNxtData(R0, r, Nh, mu, mutationKernel, tmax, dt, dtSampling, initialisation; baseFolder = "", fileAppend = "")
    
    dist = kernType(mutationKernel) * "$(std(mutationKernel))"

    dir = baseFolder * "simulations/1D/" * dist * "/dt$(dt)_dtSamp$(dtSampling)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_" * initialisation
    isdir(dir) || error("The given parameter combination `$(dir)` has not been simulated yet (or the path to the directory is incorrect, check pwd)")

    fileNxt = "Nxt" * fileAppend * ".csv"

    t, xt, Nt = Vector.(eachcol(CSV.read(joinpath(dir, fileNxt), CSV.Tables.matrix)))
    return t, xt, Nt
end

function loadSimulationDistributionData(R0, r, Nh, mu, mutationKernel, tmax, dt, dtSampling, initialisation; baseFolder = "", fileAppend = "")
    dist = kernType(mutationKernel) * "$(std(mutationKernel))"

    dir = baseFolder * "simulations/1D/" * dist * "/dt$(dt)_dtSamp$(dtSampling)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_" * initialisation
    isdir(dir) || error("The given parameter combination `$(dir)` has not been simulated yet (or the path to the directory is incorrect, check pwd)")

    filexnx = "xnx" * fileAppend * ".csv"
    filehx = "hx" * fileAppend * ".csv"

    xnx = CSV.read(joinpath(dir, filexnx), CSV.Tables.matrix)
    x = xnx[:, 1]
    nx = transpose(xnx[:, 2:end])
    hx = transpose(CSV.read(joinpath(dir, filehx), CSV.Tables.matrix))

    return nx, hx, x
end

function plotSimulationSummary(nx, hx, xmax, r, R0; tTransient = 100, dtSampling = 1)

    x = xmax-size(nx)[2]+1:xmax
    Nt = vec(sum(nx, dims = 2))
    xt = vec(sum(x' .* nx, dims = 2)) ./ Nt
    t = 0:dtSampling:dtSampling.*(length(Nt)-1)

    nx0 = nx[1, :]
    hx0 = hx[1, :]
    Nt0 = Nt[1]

    idxTransient = Int(tTransient / dtSampling) + 1

    (absorbedState, idxAbsorbed) = extinctionFlag(nx, x)
    if absorbedState > 0
        tAbsorbed = t[idxAbsorbed]
        maxIdx = Int(round((tAbsorbed - tTransient) / dtSampling)) + 1
        fastAbsorption = maxIdx < idxTransient
    else
        fastAbsorption = false
        maxIdx = length(t) - 2
    end

    v = (xt[3:end] .- xt[1:end-2]) ./ 2dtSampling
    vAverage = mean(v[idxTransient-1:maxIdx])
    vStd = std(v[idxTransient-1:maxIdx], mean = vAverage)
    NAverage = mean(Nt[idxTransient:maxIdx])
    NStd = std(Nt[idxTransient:maxIdx], mean = NAverage)
    hAverage = mean(hx[end,Int.(round.(xt[idxTransient:maxIdx]))])
    hStd = std(hx[end,Int.(round.(xt[idxTransient:maxIdx]))], mean = hAverage)

    # Produce the plots!
    # First plot: initial and final condition
    p0 = plot(x, nx0 ./ Nt0, colour=:lightsalmon, title="Virus-immune chasing, " * raw"$r = " * "$(r)," * raw"R_0 = " * "$(R0)" * raw"$" * (!fastAbsorption ? ",\n" * raw"$\bar{v} = " * "$(round(vAverage, sigdigits= 2))" * raw"\pm" * "$(round(vStd, sigdigits= 1))," * raw"\bar{N} = " * "$(Int(round(NAverage, sigdigits=2)))" * raw"\pm" * "$(Int(round(NStd, sigdigits= 1)))" * raw"$" * "\n" * raw"$\bar{h}=" * "$(Int(round(hAverage, sigdigits=2)))" * raw"\pm" * "$(Int(round(hStd, sigdigits= 1)))"  *  raw".$" : raw".") , ylabel=raw"Distributions", xlabel="x", top_margin=20Plots.px, label=raw"$n(x,0)/N(0)$", legend_position=:topright, fg_legend = :transparent)
    plot!(x, nx[end, :] ./ Nt[end], colour=:coral, label=raw"$n(x,T)/N(T)$")
    plot!(x, 2 * maximum(nx[end,:] ./ Nt[end]) * hx0./(NAverage/vAverage), colour=:lightsteelblue, label=raw"$\propto h(x,0)/(\bar{N}/\bar{v})$")
    plot!(x, 2 * maximum(nx[end,:] ./ Nt[end]) * hx[end, :]./(NAverage/vAverage), colour=:steelblue, label=raw"$\propto h(x,T)/(\bar{N}/\bar{v})$")

    # Second and third, size and position
    p1 = plot(t, Nt, color=:steelblue4, ylabel=raw"$N(t)$", xlabel=raw"$t$")
    hline!(p1, [NAverage], color = :black, ls = :dash, legend_position = :none)
    # hline!(p1, [NAverage + NStd], color = :gray, ls = :dash)
    # hline!(p1, [NAverage - NStd], color = :gray, ls = :dash)
    p2 = plot(t, xt, color=:coral, ylabel=raw"$\bar{x}(t)$", xlabel=raw"$t$", legend_position = :none)

    # Fourth plot: immune landscape
    p3 = plot(x[hx[end, :].!=0], hx[end, :][hx[end, :].!=0], color=:steelblue, ylabel=raw"$h(x)$", xlabel=raw"$x$")
    hline!(p3, [hAverage], color = :black, ls = :dash, legend_position = :none)
    # hline!(p3, [hAverage + hStd], color = :gray, ls = :dash)
    # hline!(p3, [hAverage - hStd], color = :gray, ls = :dash)

    # Last plot: speed
    p4 = plot(t[2:end-1], v, color = :coral, ylabel = raw"$v(t)$", xlabel = raw"$t$", xlims = (t[1], t[end]), ylims = (minimum(v[1:(fastAbsorption ? end : maxIdx)]), maximum(v[1:(fastAbsorption ? end : maxIdx)])), legend_position = :none)
    hline!(p4, [vAverage], color = :black, ls = :dash)
    # hline!(p4, [vAverage + vStd], color = :gray, ls = :dash)
    # hline!(p4, [vAverage - vStd], color = :gray, ls = :dash)

    l = @layout [a{0.5h}
        [grid(2, 2)]]
    p = plot(p0, p1, p2, p3, p4, layout=l, size=(600, 600))

    return p
end

function plotWaveShape(nx, hx, xmax, r, R0, Nh)

    x = xmax-size(nx)[2]+1:xmax

    (absorbedState, idxAbsorbed) = extinctionFlag(nx, x)
    absorbedState > 0 ? idx = idxAbsorbed/2 : idx = size(nx)[1]

    waveRange = max(findfirst(nx[idx, :] .> 0)-5,1):min(findlast(nx[idx, :] .> 0)+5,length(x))
    
    rInt = Int(r)
    H(x) = exp.(-abs.(x)/r)
    Hkernel = H(-5*rInt:5*rInt)
    HkernelHalfLength::Int = floor(length(Hkernel)/2)
    c = conv(hx[idx, :], Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
    f =   R0 .* exp.(-c ./ Nh) .- 1

    p1 = plot(x[waveRange], hx[idx, waveRange] ./ Nh, colour=:steelblue, xlabel=raw"$x$", label=raw"$h(x,t)/N_h$", legend_position=:right, leftmarign = 20Plots.pt, rightmargin = 20Plots.pt, lw=2, frame = :semi)
    plot!(x[waveRange], nx[idx, waveRange] ./ sum(nx[idx,:]), colour=:orangered, ylabel = raw"Densities", lw=2, label=raw"$n(x,t)/N(t)$")
    plot!([], [], label=raw"$f(x,t)/R_0$", colour=:lightsalmon, lw = 2)
    plot!(twinx(), x[waveRange], f[waveRange] ./ R0, colour=:lightsalmon, lw=2, ylabel = raw"$f(x,t)/R_0$", frame = :semi)
    plot!(widen = :false)
    hline!([ylims(p1)[2]], lc=:black, lw=1.5)

    p = plot(p1, foreground_color_legend = nothing)
    display(p)

    return p

end

function animateSimulation(nx, hx, x, Nh)
    
    (absorbedState, idxAbsorbed) = extinctionFlag(nx, x)

    animation = @animate for i in 1:(absorbedState > 0 ? min(idxAbsorbed + 100, size(nx)[1])  : size(nx)[1])
        p = plot(x, nx[i, :], colour=:coral, ylims=[0, maximum(nx)], ylabel=raw"Viral density", xlabel=raw"$x$", label = "")
        plot!(twinx(), x, hx[i, :] ./ Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"Immune memories", ylims = [0, 1], label = "")
        plot!([], [], color = :coral, label = raw"$n(x,t)$")
        plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
    end

    g = gif(animation)
    display(g)

    return g
end