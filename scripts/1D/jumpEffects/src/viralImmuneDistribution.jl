using Distributions, Plots, StatsBase, NLsolve, SpecialFunctions, DSP
import StatsBase: std, params
import Base: broadcastable

# ====================================================================
#                   New types
# ====================================================================

struct piecewiseKernel
    nonLocalMutProb::Real
    nonLocalJump::Int
    localKernel::Distribution{Univariate,Continuous}
end

function broadcastable(mutKern::piecewiseKernel)
    return Ref(mutKern)
end

function params(mutKernel::piecewiseKernel)
    (mutKernel.nonLocalMutProb, mutKernel.nonLocalJump, params(mutKernel.localKernel))
end

function std(mutKernel::piecewiseKernel)
    return std(mutKernel.localKernel)*(1-mutKernel.nonLocalMutProb)
end

struct modelParams
    r::Real
    R0::Real
    s::Real
    Nh::Real
    mutationRate::Real
    mutationKernel::Union{Distribution{Univariate,Continuous},Distribution{Univariate,Discrete},piecewiseKernel}
    D::Real
    Hkernel::Vector{<:Real}
    HkernelHalfLength::Int
end

"""
    modelParams(r, R0, Nh, mu, mutationKernel)::modelParams

Public constructor for the model parameters struct, with cross-reactivity radius `r`, basic reproduction number `R0`, host population size `Nh`, mutation rate `mu` and mutation jump distribution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function modelParams(r, R0, Nh, mu, mutationKernel)::modelParams
    s = (r > 0 ? log(R0)/r : 0)
    D = mu * std(mutationKernel)^2 / 2

    H(x) = exp.(-abs.(x)/r)
    r == 0 ? (Hkernel = [1]) : (Hkernel = H((-5*ceil(r)):(5*ceil(r))))
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    return modelParams(r, R0, s, Nh, mu, mutationKernel, D, Hkernel, HkernelHalfLength)
end

struct simulationConfig
    tmax::Real
    dt::Real
    dtSampling::Real
    idxSampling::Real
end

"""
    simulationConfig(tmax::Real, dt::Real = 0.1, dtSampling::Real = 1)::simulationConfig

Public constructor for the simulation configuration struct, with total simulation time `tmax`, simulation time increment `dt` and sampling time increment `dtSampling`.

# Examples
```julia-repl
julia>
```
"""
function simulationConfig(tmax::Real, dt::Real=0.1, dtSampling::Real=1)::simulationConfig
    idxSampling = round(Int, dtSampling/dt)
    return simulationConfig(tmax, dt, dtSampling, idxSampling)
end

mutable struct sampledResults
    nxt::Vector{Vector{Integer}}
    hxt::Vector{Vector{Integer}}
    Refft::Vector{Vector{Float64}}
end

"""
    sampledResults()::sampledResults

Public constructor for the sampled simulation results struct, created with empty vectors in fields `nxt` and `hxt` that will hold the sampled viral and immune fields respectively.

# Examples
```julia-repl
julia>
```
"""
function sampledResults()::sampledResults
    nxt = Vector{Integer}[]
    hxt = Vector{Integer}[]
    Refft = Vector{Float64}[]
    return sampledResults(nxt, hxt, Refft)
end

mutable struct macroscopicSampledResults
    Nt::Vector{Integer}
    xt::Vector{Float64}
    sigmat::Vector{Float64}
    xTt::Vector{Integer}
end

"""
    macroscopicSampledResults()::macroscopicSampledResults

Public constructor for the macroscopic sampled simulation results struct, created with empty vectors in that will hold the sampled macroscopic values.

# Examples
```julia-repl
julia>
```
"""
function macroscopicSampledResults()::macroscopicSampledResults
    Nt = Integer[]
    xt = Float64[]
    sigmat = Float64[]
    xTt = Integer[]
    return macroscopicSampledResults(Nt, xt, sigmat, xTt)
end

mutable struct viralImmuneDistribution
    space::UnitRange{<:Integer}
    nx::Vector{Integer}
    hx::Vector{Integer}
    Reff::Vector{<:Real}
end

"""
    viralImmuneDistribution(x::UnitRange{<:Integer})::viralImmuneDistribution

Produces an instance of viralImmuneDistribution population with space vector `x` and empty viral distribution `nx` and immune distribution `hx`.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution(x::UnitRange{<:Integer})::viralImmuneDistribution
    return viralImmuneDistribution(x, zeros(Int, size(x)), zeros(Int, size(x)), ones(Float64, size(x)))
end

"""
    viralImmuneDistribution(x::UnitRange{<:Integer}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution

Produces an instance of viralImmuneDistribution population with space vector `x`, viral distribution `nx`, and immune distribution `hx`.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution(x::UnitRange{<:Integer}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution
    return viralImmuneDistribution(x, nx, hx, ones(Float64, length(x)))
end

"""
    getGrowthRate!(viDist::viralImmuneDistribution, mParams::modelParams)::Vector{<:Real}

Calculates, updates in `viDist`, and returns the effective growth rate of the viral population based on the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getGrowthRate!(viDist::viralImmuneDistribution, mParams::modelParams)::Vector{<:Real}

    c = conv(viDist.hx, mParams.Hkernel)[(mParams.HkernelHalfLength+1):(end-mParams.HkernelHalfLength)]
    Reff = mParams.R0 .* exp.(-c ./ mParams.Nh)
    return viDist.Reff = Reff
end

"""
    getGrowthRateUpdate!(viDist::viralImmuneDistribution, mParams::modelParams, hGrowth::Vector{<:Integer})::Vector{<:Real}

Calculates an update to the effective growth rate based on the new memmories added, assigns it to `viDist.Reff`, and returns it, using the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getGrowthRateUpdate!(viDist::viralImmuneDistribution, mParams::modelParams, hxGrowth::Vector{<:Integer})::Vector{<:Real}

    c = idealConv(hxGrowth, mParams.Hkernel)[(mParams.HkernelHalfLength+1):(end-mParams.HkernelHalfLength)]
    return viDist.Reff = viDist.Reff .* exp.(-c ./ mParams.Nh)
end


"""
    idealConv(N, H)

Calculates a direct convolution optimized for a matrix `N` with mostly zeros (not necessarily sparse) with a kernel `H` asumed mostly .> 0. 
"""
function idealConv(N, H)
    out = zeros(length(N) .+ length(H) .- 1)
    lengthH = length(H)
    for i in axes(N, 1)
        @inbounds c = N[i]
        if c != 0
            @views out[i:(i+lengthH-1)] .+= c .* H
        end
    end
    return out
end

"""
   getImmuneUpdate!(viDist::viralImmuneDistribution, mParams::modelParams, hGrowth::Vector{<:Integer})

Updates both the effective growth rate and the immune distribution based on the new memmories added and assigns it to `viDist` atributes, using the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getImmuneUpdate!(viDist::viralImmuneDistribution, mParams::modelParams, hxGrowth::Vector{<:Integer})

    getGrowthRateUpdate!(viDist::viralImmuneDistribution, mParams::modelParams, hxGrowth::Vector{<:Integer})
    viDist.hx += hxGrowth
end

"""
   reproductionStep!(viDist::viralImmuneDistribution, simSet::simulationConfig)::Vector{<:Integer}

Performs a reproduction simulation step in the viral distribution `viDist`, following `viDist.Reff` and the timestep in `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function reproductionStep!(viDist::viralImmuneDistribution, simSet::simulationConfig)::Vector{<:Integer}

    nxGrowth = zero(viDist.nx)
    @views nxGrowth[viDist.nx .> 0] = rand.(Poisson.(viDist.Reff[viDist.nx .> 0] .* viDist.nx[viDist.nx .> 0] .* simSet.dt))
    nxGrowth = clamp.(nxGrowth, zero(nxGrowth), viDist.nx)

    viDist.nx += nxGrowth
    return nxGrowth
end

"""
   mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer})::Bool

Mutates the viral distribution in `viDist` with the mutations given by `nxMutated`, following `mParams.mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer})::Bool

    nonLocalFlag = false
    viralJumps = zero(nxMutated)

    idxMutation = findall(x -> x > 0, nxMutated)
    for idx in idxMutation
        flag = mutateNVirusAt!(viDist, mParams.mutationKernel, nxMutated[idx], idx, viralJumps)
        nonLocalFlag |= flag
    end

    viDist.nx += viralJumps - nxMutated
    return nonLocalFlag
end

"""
   mutateNVirusAt!(viDist::viralImmuneDistribution, mParams::modelParams, N::Integer, idx::Integer)::Tuple{Vector{<:Integer}, Bool}

Mutates `N` virus from position `idx` to positions sampled from `mParams.mutationKernel` in `viDist`, returning a vector with the mutated viral density. CAUTION: does not update `viDist.nx`

# Examples
```julia-repl
julia>
```
"""
function mutateNVirusAt!(viDist::viralImmuneDistribution, mutationKernel::Union{Distribution{Univariate,Continuous},Distribution{Univariate,Discrete},piecewiseKernel}, N::Integer, idx::Integer, viralJumps::Vector{<:Integer})::Bool

    numVirus = viDist.nx[idx]

    numVirus == 0 && throw(ArgumentError("There is no virus at index $idx"))

    N > 0 || throw(ArgumentError("The number of viruses to mutate must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to mutate ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    mutDisplacements, nonLocalFlag = getMutationDisplacements(mutationKernel, N)
    newIndices = round.(Int, mutDisplacements .+ idx)
    clamp!(newIndices, 1, length(viDist.space))

    [viralJumps[idx] += 1 for idx in newIndices]

    return nonLocalFlag
end

"""
   getMutationDisplacements(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)::Tuple{Vector{<:Real}, Bool}

Returns `N` displacements sampled from distibution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements(mutationKernel::Union{Distribution{Univariate,Continuous},Distribution{Univariate,Discrete}}, N::Integer)::Tuple{Vector{<:Real},Bool}
    return Distributions.rand!(mutationKernel, zeros(N)), false
end

"""
   getMutationDisplacements(mutationKernel::piecewiseKernel, Distribution{Univariate, Discrete}}, N::Integer)::Tuple{Vector{<:Real}, Bool}

Returns `N` displacements sampled from the picewise distribution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements(mutationKernel::piecewiseKernel, N::Integer)::Tuple{Vector{<:Real},Bool}
    nNonLocal = rand(Binomial(N, mutationKernel.nonLocalMutProb))
    return [Distributions.rand!(mutationKernel.localKernel, zeros(N - nNonLocal)); mutationKernel.nonLocalJump .* ones(nNonLocal)], nNonLocal > 0
end

"""
    mutationStep!(viDist::viralImmuneDistribution, mParams::modelParams,simSet::simulationConfig)::Vector{<:Integer}

Performs a mutation simulation step in the viral distribution `viDist`, with homogenous rate `mParams.mutationRate`, jumps sampled from `mParams.mutationKernel` and timestep `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function mutationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Tuple{Vector{<:Integer},Bool}

    nxMutated = zero(viDist.nx)
    @views nxMutated[viDist.nx .> 0] = rand.(Poisson.(mParams.mutationRate .* viDist.nx[viDist.nx .> 0] .* simSet.dt))
    nxMutated = clamp.(nxMutated, zero(nxMutated), viDist.nx)

    nonLocalFlag = mutateViralDistribution!(viDist, mParams, nxMutated)
    return nxMutated, nonLocalFlag
end

"""
   deathStep!(viDist::viralImmuneDistribution, simSet::simulationConfig)::Vector{<:Integer}

Performs a death simulation step in the viral distribution `viDist`, with homogenous rate 1 and timestep `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function deathStep!(viDist::viralImmuneDistribution, simSet::simulationConfig)::Vector{<:Integer}

    nxDeath = zero(viDist.nx)
    @views nxDeath[viDist.nx .> 0] = rand.(Poisson.(viDist.nx[viDist.nx .> 0] .* simSet.dt))
    nxDeath = clamp.(nxDeath, zero(nxDeath), viDist.nx)

    viDist.nx -= nxDeath
    return nxDeath
end

# Initial condition obtention

function getSteadyStateEstimate(mParams::modelParams)

    N, sigma, v = getSteadyStateLinearFitness(mParams)
    # TODO: think about the FKPP limit :(

    N = max(N, 1000)
    sigma = max(sigma, 1)

    return N, sigma, v
end

function getSteadyStateLinearFitness(mParams::modelParams)
    v0 = (mParams.D^2 * mParams.s * log(mParams.Nh * (mParams.D * mParams.s^2)^(1/3)))^(1/3)
    N0 = round(mParams.Nh * v0 * mParams.s)

    func2zero = x -> linearFitnessvNEquation(x, mParams.Nh, mParams.s, mParams.D)
    (v0, N0) = nlsolve(func2zero, [v0, N0]).zero
    sigma0 = sqrt(v0 / mParams.s)

    return N0, sigma0, v0
end

function linearFitnessvNEquation(vN, Nh, s, D)
    l1 = vN[2] - Nh*s*vN[1]
    l2 = vN[1] - (max(D^2 * s * log(vN[2] * (D * s^2)^(1/3)), 0))^(1/3)

    return [l1, l2]
end

isTooCloseToBoundary(viDist::viralImmuneDistribution, mutationKernel::Union{Distribution{Univariate,Continuous},Distribution{Univariate,Discrete}}) = findlast(viDist.nx .> 0) + std(mutationKernel)*3 > length(viDist.space)

isTooCloseToBoundary(viDist::viralImmuneDistribution, mutationKernel::piecewiseKernel) = findlast(viDist.nx .> 0) + max(std(mutationKernel)*3, mutationKernel.nonLocalJump) > length(viDist.space)

## =========================================================
#                   Public functions
#  =========================================================

function getInitialDistribution!(viDist::viralImmuneDistribution, mParams::modelParams)

    N0, sigma0, v0 = getSteadyStateEstimate(mParams)
    H0 = mParams.Nh * mParams.s

    gaussianCond(x, sig) = exp(-x^2/(2*sig^2))/sqrt(2pi*sig^2)

    x = viDist.space

    nx0 = round.(Int, N0 .* gaussianCond.(x, sigma0))
    hx0 = round.(Int, H0 .* (1 .+ erf.(.-x ./ sqrt(2*sigma0^2))) .* (x .> -mParams.HkernelHalfLength) ./ 2)

    viDist.nx = nx0
    viDist.hx = hx0

    getGrowthRate!(viDist, mParams)
end

function getInitialDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, sampDist::sampledResults)

    getInitialDistribution!(viDist::viralImmuneDistribution, mParams::modelParams)
    sampleDistributions!(viDist, sampDist)
end

function getInitialDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, sampMacro::macroscopicSampledResults)

    getInitialDistribution!(viDist::viralImmuneDistribution, mParams::modelParams)
    sampleMacroscopicObservables!(viDist, sampMacro)
end

# Simulation

"""
   simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Tuple{Bool, Bool}

Performs a simulation step in the viral distribution `viDist`, with parameters `mParams`, timestep `simSet.dt`, and returns the extinction state (true if not extinct) and whether a non-local event has happened or not in the last step.

# Examples
```julia-repl
julia>
```
"""
function simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Tuple{Bool,Bool}

    reproductionStep!(viDist, simSet)
    _, nonLocalFlag = mutationStep!(viDist, mParams, simSet)
    hxGrowth = deathStep!(viDist, simSet)
    getImmuneUpdate!(viDist, mParams, hxGrowth)

    return !iszero(viDist.nx), nonLocalFlag
end

isTooCloseToBoundary(viDist::viralImmuneDistribution, mParams::modelParams) = isTooCloseToBoundary(viDist, mParams.mutationKernel)

"""
   translateDistributionBackLeft!(viDist::viralImmuneDistribution, mParams::modelParams)

Translates the viral distribution as close to the begining of the `viDist.space` vector, keeping a `mParams.HkernelHalfLength` distance before the first virus to keep the accumulated effect of the immune memmories.

# Examples
```julia-repl
julia>
```
"""
function translateDistributionBackLeft!(viDist::viralImmuneDistribution, mParams::modelParams)

    idxWave = findall(viDist.nx .> 0)
    firstIdx = mParams.HkernelHalfLength
    deltaIdx = first(idxWave) - firstIdx

    if deltaIdx > 0
        newIdx = idxWave .- deltaIdx
    elseif maximum(idxWave) > 2*firstIdx
        deltaIdx = firstIdx
        newIdx = idxWave .- deltaIdx
        filter!(x -> x >= firstIdx, newIdx)
        idxWave = newIdx .+ deltaIdx
    else
        deltaIdx = 0
        newIdx = idxWave .- deltaIdx
    end


    newNx = zero(viDist.nx)
    newNx[newIdx] = viDist.nx[idxWave]

    newHx = zero(viDist.hx)
    newHx[1:last(newIdx)] = viDist.hx[(1+deltaIdx):last(idxWave)]

    newSpace = viDist.space .+ deltaIdx

    viDist.nx = newNx
    viDist.hx = newHx
    viDist.space = newSpace

    getGrowthRate!(viDist, mParams)
end

# Sampling

"""
    isTime2Sample(t::Real, simSet::simulationConfig)::Bool

Returns true if the time `t` is a multiple of simSet.dtSampling (and thus time to save a sample of the distributions).
"""
function isTime2Sample(t::Real, simSet::simulationConfig)::Bool
    return (round(Int, t / simSet.dt) % simSet.idxSampling) == 0
end

"""
    sampleDistributions(viDist::viralImmuneDistribution, sampDist::sampledResults)

Samples the viral and immune distribution in `viDist` by pushing them to the corresponding vectors in `sampDist`.
"""
function sampleDistributions!(viDist::viralImmuneDistribution, sampDist::sampledResults)
    push!(sampDist.nxt, viDist.nx)
    push!(sampDist.hxt, viDist.hx)
    push!(sampDist.Refft, viDist.Reff)
end

function sampleMacroscopicObservables!(viDist::viralImmuneDistribution, sampMacro::macroscopicSampledResults)
    N = sum(viDist.nx)
    xn = sum(viDist.nx .* viDist.space) ./ N
    sigma = sum(viDist.nx .* viDist.space .^ 2) ./ N - xn .^ 2
    xT = viDist.space[findlast(viDist.nx .> 0)]

    push!(sampMacro.Nt, N)
    push!(sampMacro.xt, xn)
    push!(sampMacro.sigmat, sigma)
    push!(sampMacro.xTt, xT)
end

# ============================================================
#                       Plotting tools
#  ============================================================

function plotAnimation(sampDist::sampledResults, x::Union{UnitRange,Nothing}=nothing)

    isempty(sampDist.nxt) && throw(ArgumentError("The given sampled distributions are empty"))

    isnothing(x) && (x = 1:length(first(sampDist.nxt)))

    maxN = maximum(hcat(sampDist.nxt...))
    maxH = maximum(hcat(sampDist.hxt...))
    NscalingFactor = maxH/maxN

    animation = @animate for i in 1:length(sampDist.nxt)
        p = Plots.plot(x, replace(sampDist.hxt[i], 0 => NaN), ylabel=raw"$h(x,y)$", xlabel=raw"$x$", c=:steelblue, ylims=(0, maxH))
        Plots.plot!(p, x, replace(sampDist.nxt[i], 0 => NaN) .* NscalingFactor, ylabel=raw"$n(x,y)$", c=:coral)
    end

    g = gif(animation)
    display(g)
    return g
end