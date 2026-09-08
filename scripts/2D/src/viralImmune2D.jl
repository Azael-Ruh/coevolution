using StatsBase, Distributions, DSP, LinearAlgebra, NLsolve, SpecialFunctions, Plots, Random
import StatsBase: std, params
import Base: broadcastable, length

include("./speciationTools.jl")

# ========================================================
#           Type definitions and constructors
# ========================================================

struct piecewiseKernel2D
    nonLocalMutProb::Real
    nonLocalJump::Int
    localKernel::Union{Distribution{Univariate, Continuous}, Distribution{Multivariate, Continuous}, Distribution{Univariate, Discrete}, Distribution{Multivariate, Discrete}}
end

function broadcastable(mutKern::piecewiseKernel2D)
    return Ref(mutKern)
end

function params(mutKernel::piecewiseKernel2D)
    (mutKernel.nonLocalMutProb, mutKernel.nonLocalJump, params(mutKernel.localKernel))
end

function std(mutKernel::piecewiseKernel2D)
    return std(mutKernel.localKernel)*(1-mutKernel.nonLocalMutProb)
end

struct modelParams
    r::Real
    R0::Real
    s::Real
    Nh::Real
    mutationRate::Real
    mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}, Distribution{Multivariate, Discrete}, Distribution{Multivariate, Continuous}, piecewiseKernel2D}
    D::Real
    Hkernel::Matrix{<:Real}
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
function modelParams(r, R0, Nh, mu, mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}, piecewiseKernel2D})::modelParams
    s = (r > 0 ? log(R0)/r : 2log(R0))
    D = mu * std(mutationKernel)[1]^2 / 2

    H(x) = exp.(-abs.(x)/r)
    r == 0 ? (Hkernel = ones(1,1)) : begin
        HkernelSpace = -5*ceil(r):5*ceil(r)
        Hkernel = H(sqrt.(HkernelSpace.^2 .+ HkernelSpace'.^2))
    end
    HkernelHalfLength::Int = floor(size(Hkernel)[1]/2)

    return modelParams(r, R0, s, Nh, mu, mutationKernel, D, Hkernel, HkernelHalfLength)
end

#TODO: try with pre-made multivariate distributions
"""
    modelParams(r, R0, Nh, mu, mutationKernel)::modelParams

Public constructor for the model parameters struct, with cross-reactivity radius `r`, basic reproduction number `R0`, host population size `Nh`, multivariate mutation rate `mu` and mutation jump distribution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function modelParams(r, R0, Nh, mu, mutationKernel::Union{Distribution{Multivariate, Continuous}, Distribution{Multivariate, Discrete}})::modelParams
    s = (r > 0 ? log(R0)/r : 2log(R0))
    D = mu * std(mutationKernel)[1]^2 / 2

    H(x) = exp.(-abs.(x)/r)
    r == 0 ? (Hkernel = ones(1,1)) : begin
        HkernelSpace = -5*ceil(r):5*ceil(r)
        Hkernel = H(sqrt.(HkernelSpace.^2 .+ HkernelSpace'.^2))
    end
    HkernelHalfLength::Int = floor(size(Hkernel)[1]/2)

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
function simulationConfig(tmax::Real, dt::Real = 0.1, dtSampling::Real = 1)::simulationConfig
    idxSampling = round(Int, dtSampling/dt)
    return simulationConfig(tmax, dt, dtSampling, idxSampling)
end

mutable struct sampledResults
    nxt::Vector{Matrix{Integer}}
    hxt::Vector{Matrix{Integer}}
    speciest::Vector{Vector{viralSpecies}}
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
    nxt = Matrix{Integer}[]
    hxt = Matrix{Integer}[]
    speciest = Vector{viralSpecies}[]
    return sampledResults(nxt, hxt, speciest)
end

mutable struct viralImmuneDistribution2D
    x::UnitRange{Int}
    y::UnitRange{Int}
    origin::CartesianIndex{2}
    lowerIdxLimit::CartesianIndex{2}
    higherIdxLimit::CartesianIndex{2}
    nx::Matrix{Int}
    hx::Matrix{Int}
    Reff::Matrix{Real}
end

"""
    viralImmuneDistribution2D(x::UnitRange{Int}, y::UnitRange{Int}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution2D

Produces an instance of viralImmuneDistribution2D population with space vectors `x`, `y`, viral distribution `nx`, immune distribution `hx` and null net grwht.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution2D(x::UnitRange{Int}, y::UnitRange{Int}, nx::Matrix{<:Integer}, hx::Matrix{<:Integer})::viralImmuneDistribution2D
    (length(x), length(y)) == size(nx) == size(hx) || throw(DimensionMismatch("The provided arguments do not have matching dimensionality."))
    origin = CartesianIndex(findfirst(x .== 0), findfirst(y .== 0))
    spaceIdxs = CartesianIndices(nx)
    lowIdx = minimum(spaceIdxs)
    highIdx = maximum(spaceIdxs)
    return viralImmuneDistribution2D(x, y, origin, lowIdx, highIdx, nx, hx, ones(Float64, size(nx)))
end

"""
    viralImmuneDistribution2D(x::UnitRange{Int}, y::UnitRange{Int})::viralImmuneDistribution2D

Produces an instance of viralImmuneDistribution2D population with space vectors `x` and `y`, and null viral and immune distribution and net growht.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution2D(x::UnitRange{Int}, y::UnitRange{Int})::viralImmuneDistribution2D
    nx = zeros(Int, length(x), length(y))
    hx = zeros(Int, length(x), length(y))
    origin = CartesianIndex(findfirst(x .== 0), findfirst(y .== 0))
    spaceIdxs = CartesianIndices(nx)
    lowIdx = minimum(spaceIdxs)
    highIdx = maximum(spaceIdxs)
    return viralImmuneDistribution2D(x, y, origin, lowIdx, highIdx, nx, hx, ones(Float64, size(nx)))
end

# =======================================================
#               Private functions
# =======================================================

"""
    getGrowthRate!(vi2D::viralImmuneDistribution, mParams::modelParams)::Vector{<:Real}

Calculates, updates in `vi2D`, and returns the effective growth rate of the viral population based on the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getGrowthRate!(vi2D::viralImmuneDistribution2D, mParams::modelParams)::Matrix{<:Real}
    
    c = conv(vi2D.hx, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength, mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
    Reff = mParams.R0 .* exp.(-c ./ mParams.Nh)
    return vi2D.Reff = Reff
end

"""
    getGrowthRateUpdate!(vi2D::viralImmuneDistribution2D, mParams::modelParams, hGrowth::Matrix{<:Integer})::Matrix{<:Real}

Calculates an update to the effective growth rate based on the new memmories added, assigns it to `vi2D.Reff`, and returns it, using the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getGrowthRateUpdate!(vi2D::viralImmuneDistribution2D, mParams::modelParams, hxGrowth::Matrix{<:Integer})::Matrix{<:Real}
    
    c = conv(hxGrowth, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength, mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
    return vi2D.Reff = vi2D.Reff .* exp.(-c ./ mParams.Nh)
end

"""
   getImmuneUpdate!(vi2D::viralImmuneDistribution2D, mParams::modelParams, hxGrowth::Matrix{<:Integer})

Updates both the effective growth rate and the immune distribution based on the new memmories added and assigns it to `vi2D` atributes, using the parameters `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getImmuneUpdate!(vi2D::viralImmuneDistribution2D, mParams::modelParams, hxGrowth::Matrix{<:Integer})

    getGrowthRateUpdate!(vi2D, mParams, hxGrowth)
    vi2D.hx += hxGrowth
end

"""
   reproductionStep!(vi2D::viralImmuneDistribution2D, simSet::simulationConfig)::Matrix{<:Integer}

Performs a reproduction simulation step in the viral distribution `vi2D`, following `vi2D.Reff` and the timestep in `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function reproductionStep!(vi2D::viralImmuneDistribution2D, simSet::simulationConfig)::Matrix{<:Integer}

    nxGrowth = rand.(Poisson.(vi2D.Reff .* vi2D.nx .* simSet.dt))
    nxGrowth = clamp.(nxGrowth, zero(nxGrowth), vi2D.nx)
    vi2D.nx += nxGrowth
    return nxGrowth
end

"""
   deathStep!(vi2D::viralImmuneDistribution2D, simSet::simulationConfig)::Matrix{<:Integer}

Performs a death simulation step in the viral distribution `vi2D`, with homogenous rate 1 and timestep `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function deathStep!(vi2D::viralImmuneDistribution2D, simSet::simulationConfig)::Matrix{<:Integer}

    nxDeath = rand.(Poisson.(vi2D.nx .* simSet.dt))
    nxDeath = clamp.(nxDeath, zero(nxDeath), vi2D.nx)
    vi2D.nx -= nxDeath
    return nxDeath
end

"""
    mutationStep!(vi2D::viralImmuneDistribution2D, mParams::modelParams,simSet::simulationConfig)::Bool

Performs a mutation simulation step in the viral distribution `vi2D`, with homogenous rate `mParams.mutationRate`, jumps sampled from `mParams.mutationKernel` and timestep `simSet.dt`. Returns a flag for non-local events.

# Examples
```julia-repl
julia>
```
"""
function mutationStep!(vi2D::viralImmuneDistribution2D, mParams::modelParams,simSet::simulationConfig)::Bool

    nxMutated = rand.(Poisson.(mParams.mutationRate .* vi2D.nx .* simSet.dt))
    nxMutated = clamp.(nxMutated, zero(nxMutated), vi2D.nx)
    nonLocalFlag = mutateViralDistribution!(vi2D, mParams, nxMutated)
    return nonLocalFlag
end

"""
   mutateViralDistribution!(vi2D::viralImmuneDistribution2D, mParams::modelParams, nxMutated::Matrix{<:Integer})::Bool

Mutates the viral distribution in `vi2D` with the mutations given by `nxMutated`, following `mParams.mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function mutateViralDistribution!(vi2D::viralImmuneDistribution2D, mParams::modelParams, nxMutated::Matrix{<:Integer})::Bool

    nonLocalFlag = false
    viralJumps = zero(nxMutated)

    idxGrowth = findall(x -> x > 0, nxMutated)
    for idx in idxGrowth
        jump, flag = getNJumpsAt(vi2D, mParams.mutationKernel, nxMutated[idx], idx)
        viralJumps += jump
        nonLocalFlag |= flag
    end

    vi2D.nx += viralJumps - nxMutated
    return nonLocalFlag
end

"""
   getNJumpsAt(vi2D::viralImmuneDistribution2D, mParams::modelParams, N::Integer, idx::CartesianIndex)::Tuple{Matrix{<:Integer}, Bool}

Mutates `N` virus from position `idx` to positions sampled from `mParams.mutationKernel` in `vi2D`, returning a matrix with the mutated viral density. If a non-local jump is found, it sets nonLocalJumpFlag to true CAUTION: does not update `vi2D.nx`

# Examples
```julia-repl
julia>
```
"""
function getNJumpsAt(vi2D::viralImmuneDistribution2D, mutationKernel::Union{Distribution{Multivariate, Discrete}, Distribution{Multivariate, Continuous}, Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}, piecewiseKernel2D}, N::Integer, idx::CartesianIndex)::Tuple{Matrix{<:Integer}, Bool}

    numVirus = vi2D.nx[idx]

    numVirus == 0 && throw(ArgumentError("There is no virus at index $idx"))

    N > 0 || throw(ArgumentError("The number of viruses to mutate must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to mutate ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    mutDisplacements, nonLocalFlag = getMutationDisplacements2D(mutationKernel, N)
    newIndices = filter(disp -> checkbounds(Bool, vi2D.nx, disp), mutDisplacements .+ idx)

    mutJumps = zero(vi2D.nx)
    [mutJumps[idx] += 1 for idx in newIndices]

    return mutJumps, nonLocalFlag
end

"""
   getMutationDisplacements2D(mutationKernel::piecewiseKernel2D, N::Integer, nonLocalJumpFlag::Ref{Bool})::Tuple{Vector{CartesianIndex{2}}, Bool}

Returns `N` displacements sampled from piecewise distibution `mutationKernel`. If any non-local jump is found

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements2D(mutationKernel::piecewiseKernel2D, N::Integer)::Tuple{Vector{CartesianIndex{2}}, Bool}
    NnonLocal = sum(rand(N) .<= mutationKernel.nonLocalMutProb)
    
    localIdxs, _ = getMutationDisplacements2D(mutationKernel.localKernel, N - NnonLocal)
    
    nonLocalJumpAngles = 2pi .* rand(NnonLocal)
    nonLocalIdxs = [CartesianIndex(round.(Int, mutationKernel.nonLocalJump .* [cos(theta), sin(theta)])...) for theta in nonLocalJumpAngles]
    
    return vcat(localIdxs, nonLocalIdxs), NnonLocal > 0
end

"""
   getMutationDisplacements2D(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)::Tuple{Vector{CartesianIndex{2}}, Bool}

Returns `N` displacements sampled from distibution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements2D(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)::Tuple{Vector{CartesianIndex{2}}, Bool}
    return [CartesianIndex(round.(Int, rand(mutationKernel, 2))...) for i in 1:N], false
end

"""
   getMutationDisplacements2D(mutationKernel::Union{Distribution{Multivariate, Continuous}, Distribution{Multivariate, Discrete}}, N::Integer)::Tuple{Vector{CartesianIndex{2}}, Bool}

Returns `N` displacements sampled from multivariate distibution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements2D(mutationKernel::Union{Distribution{Multivariate, Continuous}, Distribution{Multivariate, Discrete}}, N::Integer)::Tuple{Vector{CartesianIndex{2}}, Bool}
    return [CartesianIndex(round.(Int, rand(mutationKernel))...) for i in 1:N], false
end

function getIdxLimits(distribution::Matrix{<:Real})::Tuple{CartesianIndex, CartesianIndex}
    idxWave = findall(distribution .> 0)
    return minimum(idxWave), maximum(idxWave)
end

function clampIdx(idx::CartesianIndex, idxMin::CartesianIndex, idxMax::CartesianIndex)::CartesianIndex
    return min(max(idx, idxMin), idxMax)
end

function getCentralIdx(minIdx::CartesianIndex, maxIdx::CartesianIndex)::CartesianIndex
    return CartesianIndex([abs((maxIdx[i] + minIdx[i]) ÷ 2) for i in 1:length(minIdx)]...)
end

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

function getPlottingIndexes(dist::Matrix{<:Integer})::CartesianIndices
    minIdx, maxIdx = getIdxLimits(dist)
    return minIdx:maxIdx
end

# =======================================================
#                   Public functions
# =======================================================


"""
   simulationStep!(vi2D::viralImmuneDistribution2D, mParams::modelParams, simSet::simulationConfig)::Bool

Performs a simulation step in the viral distribution `vi2D`, with parameters `mParams`, timestep `simSet.dt`, and returns the extinction state (true if not extinct) of the distribution.

# Examples
```julia-repl
julia>
```
"""
function simulationStep!(vi2D::viralImmuneDistribution2D, mParams::modelParams, simSet::simulationConfig)::Tuple{Bool, Bool}

    reproductionStep!(vi2D, simSet)
    nonLocalFlag = mutationStep!(vi2D, mParams, simSet)
    hxGrowth = deathStep!(vi2D, simSet)
    getImmuneUpdate!(vi2D, mParams, hxGrowth)

    return iszero(vi2D.nx), nonLocalFlag
end

"""
   translateDistributionBack2Origin!(vi2D::viralImmuneDistribution2D, mParams::modelParams)

Translates the viral distribution to the origin of the space defined by  `vi2D.x` and `vi2D.y` vectors, keeping the immune system in a distance `mParams.HkernelHalfLength` around the viral cloud to keep the accumulated effect of the immune memmories.

# Examples
```julia-repl
julia>
```
"""
function translateDistributionBack2Origin!(vi2D::viralImmuneDistribution2D, mParams::modelParams)
    
    # Get the space where viruses are alive, expand it by `mParams.HkernelHalfLength` in all directions to also get the immune system that interacts with it (clamping it so it is in bounds) and creating the translated version around the origin. 
    minIdx, maxIdx = getIdxLimits(vi2D.nx)
    immuneLengthIdx = CartesianIndex(ceil(Int, 5*mParams.r), ceil(Int, 5*mParams.r))
    minIdxImmunity = clampIdx(minIdx - immuneLengthIdx, vi2D.lowerIdxLimit, vi2D.higherIdxLimit)
    maxIdxImmunity = clampIdx(maxIdx + immuneLengthIdx, vi2D.lowerIdxLimit, vi2D.higherIdxLimit)
    newMinIdx = clampIdx(minIdxImmunity - getCentralIdx(minIdx, maxIdx) + vi2D.origin, vi2D.lowerIdxLimit, vi2D.higherIdxLimit)
    newMaxIdx = clampIdx(maxIdxImmunity - getCentralIdx(minIdx, maxIdx) + vi2D.origin, vi2D.lowerIdxLimit, vi2D.higherIdxLimit)
    newIndices = newMinIdx:newMaxIdx 
    waveIndices = newIndices .+ getCentralIdx(minIdx, maxIdx) .- vi2D.origin

    checkbounds(vi2D.nx, newIndices)

    newNx = zero(vi2D.nx)
    newNx[newIndices] = vi2D.nx[waveIndices]

    newHx = zero(vi2D.hx)
    newHx[newIndices] = vi2D.hx[waveIndices]

    vi2D.nx = newNx
    vi2D.hx = newHx

    getGrowthRate!(vi2D, mParams)
end

function getInitialDistribution!( vi2D::viralImmuneDistribution2D, mParams::modelParams, sampDist::sampledResults)
        
    transverseWidthFactor = sqrt(1.66)

    N0, sigma0, v0 = getSteadyStateEstimate(mParams)
    H0 = mParams.Nh * mParams.s
    
    gaussianCond(x, sig) = exp(-x^2/(2*sig^2))/sqrt(2pi*sig^2)

    x = vi2D.x
    y = vi2D.y

    nx0 = round.(Int, N0 .* gaussianCond.(x, sigma0) .* gaussianCond.(y, transverseWidthFactor * sigma0)')
    hx0 = round.(Int, H0 .* (1 .+ erf.( .-x ./ sqrt(2*sigma0^2))) .* (x .> -5mParams.r) ./ 2 .* (gaussianCond.(y, transverseWidthFactor * sigma0))')

    vi2D.nx = nx0
    vi2D.hx = hx0

    getGrowthRate!(vi2D, mParams)
    sampleDistributionsNSpecies!(vi2D, sampDist, mParams)
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
    sampleDistributions(vi2D::viralImmuneDistribution2D, sampDist::sampledResults)

Samples the viral and immune distribution in `vi2D` by pushing them to the corresponding vectors in `sampDist`.
"""
function sampleDistributions!(vi2D::viralImmuneDistribution2D, sampDist::sampledResults)
    push!(sampDist.nxt, vi2D.nx)
    push!(sampDist.hxt, vi2D.hx)
end

function sampleDistributionsNSpecies!(vi2D::viralImmuneDistribution2D, sampDist::sampledResults, mParams::modelParams)
    push!(sampDist.nxt, vi2D.nx)
    push!(sampDist.hxt, vi2D.hx)
    push!(sampDist.speciest, getSpecies(vi2D, mParams.r))
end

# Plotting

function plotDistributions(vi2D::viralImmuneDistribution2D)::Plots.Plot
    idxs = CartesianIndices(vi2D.nx)
    return plotDistributionsIdxs(vi2D.nx, vi2D.hx, vi2D.x, vi2D.y, idxs)
end

function plotDistributionsIdxs(nx::Matrix{<:Integer}, hx::Matrix{<:Integer}, x::UnitRange, y::UnitRange, plottingIdx::CartesianIndices)::Plots.Plot    
    
    nxNaN = replace(nx, 0 => NaN)
    hxNaN = replace(hx, 0 => NaN)

    xRange = x[plottingIdx.indices[1]]
    yRange = y[plottingIdx.indices[2]]

    nFactor = maximum(hx) / maximum(nx)
    hp = heatmap(xRange, yRange, hxNaN[plottingIdx]', zlabel=raw"$h(x,y)$", xlabel=raw"$x$", ylabel = raw"$y$", c=cgrad(:Blues, scale  = :exp), colorbar_title = raw"$h(x,y)$", cbar = :none)
    return heatmap!(hp, xRange, yRange, nxNaN[plottingIdx]' .* nFactor, zlabel=raw"$n(x,y)$", xlabel=raw"$x$", ylabel = raw"$y$", c=cgrad(:magma, scale  = :exp), colorbar_title = raw"$n(x,y)$", cbar = :none)
end

function plotAnimation(sampDist::sampledResults, x::Union{UnitRange, Nothing} = nothing, y::Union{UnitRange, Nothing} = nothing)

    isempty(sampDist.nxt) && throw(ArgumentError("The given sampled distributions are empty"))

    isnothing(x) && (x = 1:size(first(sampDist.nxt))[1])
    isnothing(y) && (y = 1:size(first(sampDist.nxt))[2])

    idxs = CartesianIndices(first(sampDist.nxt))

    maxN = maximum(hcat(sampDist.nxt...))
    maxH = maximum(hcat(sampDist.hxt...))
    NscalingFactor = maxH/maxN

    animation = @animate for i in 1:length(sampDist.nxt)
        hp = heatmap(x, y, replace(sampDist.hxt[i], 0 => NaN)', zlabel=raw"$h(x,y)$", xlabel=raw"$x$", ylabel = raw"$y$", clims = (0, maxH), c=cgrad(:Blues, scale  = :exp), cbar = :none)
        heatmap!(x, y, replace(sampDist.nxt[i], 0 => NaN)' .* NscalingFactor, zlabel = raw"$n(x,y)$", c=cgrad(:magma, scale  = :exp), cbar = :none)
    end

    g = gif(animation)
    display(g)
    return g
end

# Viral species 

function getSpecies(vi2D::viralImmuneDistribution2D, r::Real; withMask::Bool = false)::Vector{viralSpecies}
    viralLabels = getViralLabels(vi2D.nx, r)
    speciesVector = getSpeciesValues(viralLabels, vi2D.nx, vi2D.x, vi2D.y, vi2D.Reff, withMask = withMask)
end

function isolateBestSpeciesBack2Origin!(vi2D::viralImmuneDistribution2D, mParams::modelParams)
    speciesVector = getSpecies(vi2D, mParams.r, withMask = true)
    bestSpecies = getBestSpecies(speciesVector)
    vi2D.nx = getMaskedDistribution(bestSpecies)
    translateDistributionBack2Origin!(vi2D, mParams)
end

function hasSpeciationHappened(sampDist::sampledResults)::Bool
    return length(sampDist.speciest[end]) > length(sampDist.speciest[end - 1])
end

function hasExtinctionHappened(sampDist::sampledResults)::Bool
    return length(sampDist.speciest[end]) < length(sampDist.speciest[end - 1])
end

# =========================================================
#                       User distributions
# =========================================================


# ============= Next-Neighbour Distribution =============
#TODO: Extend the distribution to an arbitrary dimension!

"""
Next-neighbour jump distribution with counter-clockwise probability vector p.
"""
struct NextNeighbour{T<:Real, TV<:AbstractVector{T}} <: DiscreteMultivariateDistribution
    p::TV
    NextNeighbour{T, TV}(p::TV) where {T <: Real, TV <: AbstractVector{T}} = new{T, TV}(p)
end

function NextNeighbour(p::AbstractVector{T}; check_args::Bool = true) where {T<:Real}
    length(p) == 4 || throw(ArgumentError("p is not of length 4."))
    isprobvec(p) || throw(ArgumentError("p is not a probability vector."))
    return NextNeighbour{T, typeof(p)}(p)
end

NextNeighbour() = NextNeighbour{Float64, Vector{Float64}}(0.25*ones(4))

# Parameters

ncategories(d::NextNeighbour) = length(d.p)
length(d::NextNeighbour) = Int(ncategories(d)/2)
probs(d::NextNeighbour) = d.p

params(d::NextNeighbour) = (d.p)
@inline partype(d::NextNeighbour{T}) where {T<:Real} = T

# Statistics

mean(d::NextNeighbour{T}) where T<:Real = Vector{T}([d.p[1] - d.p[3], d.p[2] - d.p[4]])

function var(d::NextNeighbour{T}) where T<:Real
    p = probs(d)
    k = length(d)

    v = Vector{T}(undef, k)
    for i = 1:k
        p_i = p[i]
        p_j = p[i+k]
        v[i] = (p_i + p_j) - (p_i - p_j)^2
    end
    v
end

std(d::NextNeighbour) = sqrt.(var(d))

function cov(d::NextNeighbour{T}) where T<:Real
    p = probs(d)
    k = length(d)

    C = Matrix{T}(undef, k, k)
    m = mean(d)
    v = var(d)

    for j = 1:k
        C[j,j] = v[j]
    end

    for j = 1:k-1
        for i = j+1:k
            C[i,j] = -m[i]*m[j]
            C[i,j] = C[j,i]
        end
    end
    C
end

function mgf(d::NextNeighbour{T}, t::AbstractVector) where T<:Real
    p = probs(d)
    k = length(d)
    s = zero(T)
    for i in 1:length(p)
        s += p[i] * exp((-1)^(floor(i/k)) * t[i%k])
    end
    return s
end

function cf(d::NextNeighbour{T}, t::AbstractVector) where T<:Real
    p = probs(d)
    k = length(d)
    s = zero(Complex{T})
    for i in 1:length(p)
        s += p[i] * exp(im * (-1)^(floor(i/k)) * t[i%k])
    end
    return s
end

entropy(d::NextNeighbour) = entropy(p.d)

# Evaluation

function insupport(d::NextNeighbour, x::AbstractVector{T}) where T<:Real
    k = length(d)
    length(x) == k || return false

    is1Dstep = sum(x .== 0) == k - 1
    isneighbour = sum(x) == 1
    return is1Dstep * isneighbour
end

function Distributions._logpdf(d::NextNeighbour, x::AbstractVector{T}) where T<:Real
    
    p = probs(d)
    S = eltype(p)
    R = promote_type(T, S)
    insupport(d,x) || return -R(Inf)
    
    return R(log(p[(2 - x[1])*abs(x[1]) + (3 - x[2])*abs(x[2])]))
end

# Sampling

function Distributions._rand!(rng::AbstractRNG, d::NextNeighbour, x::AbstractVector{<:Real})
    k = length(d)
    length(x) == k || throw(DimensionMismatch("invalid argument dimension."))

    p = probs(d)
    cump = cumsum(p)
    
    xi = rand(rng)
    neighbour = findfirst(cump .> xi)

    x[1] = (-1)^floor(neighbour / k) * (neighbour % k)
    x[2] = (-1)^floor((neighbour - 1) / k) * ((neighbour - 1) % k)

    return x
end

sampler(d::Multinomial) = d