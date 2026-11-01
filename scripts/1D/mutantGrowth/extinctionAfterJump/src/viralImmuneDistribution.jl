# module viralImmune1D

# export modelParams, simulationConfig, sampledResults, viralImmuneDistribution, newlySeededDistribution, getGrowthRateLinear!, simulationStep!, isTime2Sample, sampleDistributions!, plotConfig, animateDistributions

using StatsBase, Distributions, DSP, Plots

struct modelParams
    r::Real
    R0::Real
    s::Real
    Nh::Real
    mutationRate::Real
    mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}
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
    r == 0 ? (Hkernel = [1]) : (Hkernel = H(-5*ceil(r):5*ceil(r)))
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
function simulationConfig(tmax::Real, dt::Real = 0.1, dtSampling::Real = 1)::simulationConfig
    idxSampling = round(Int, dtSampling/dt)
    return simulationConfig(tmax, dt, dtSampling, idxSampling)
end

mutable struct viralImmuneDistribution
    space::UnitRange{<:Integer}
    nx::Vector{<:Integer}
    hx::Vector{<:Integer}
    Reff::Vector{<:Real}
end

"""
    viralImmuneDistribution(x::UnitRange{<:Integer}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution

Produces an instance of viralImmuneDistribution population with space vector `x`, viral distribution `nx` and immune distribution `hx`.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution(x::UnitRange{<:Integer}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution
    return viralImmuneDistribution(x, nx, hx, ones(Float64, length(x)))
end

mutable struct sampledResults
    space::UnitRange{<:Integer}
    nxt::Vector{Vector{<:Integer}}
    hxt::Vector{Vector{<:Integer}}
    Refft::Vector{Vector{Float64}}
end

"""
    sampledResults(viDist::viralImmuneDistribution)::sampledResults

Public constructor for the sampled simulation results struct, created with empty vectors in fields `nxt` and `hxt` that will hold the sampled viral and immune fields respectively.

# Examples
```julia-repl
julia>
```
"""
function sampledResults(viDist::viralImmuneDistribution)::sampledResults
    space = viDist.space
    nxt = Vector{Integer}[]
    hxt = Vector{Integer}[]
    Refft = Vector{Float64}[]
    return sampledResults(space, nxt, hxt, Refft)
end

"""
    newlySeededDistribution(x::UnitRange{<:Integer}, x0::Int)::viralImmuneDistribution

Produces an instance of viralImmuneDistribution population with space vector `x`, and a single virus at position `x0`.

# Examples
```julia-repl
julia>
```
"""
function newlySeededDistribution(x::UnitRange{<:Integer}, x0::Int)::viralImmuneDistribution
    nx = zeros(Int, length(x))
    nx[x .== x0] .= 1
    
    hx = zeros(Int, length(x))
    return viralImmuneDistribution(x, nx, hx, ones(Float64, length(x)))
end

"""
    getGrowthRateLinear!(viDist::viralImmuneDistribution, mParams::modelParams)::Vector{<:Real}

Calculates an initial linear growth rate for the viral distribution `viDist` following parameters in `mParams`.

# Examples
```julia-repl
julia>
```
"""
function getGrowthRateLinear!(viDist::viralImmuneDistribution, mParams::modelParams)::Vector{<:Real}
    Reff = collect(mParams.s .* viDist.space .+ 1)
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
    
    c = conv(hxGrowth, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
    return viDist.Reff = viDist.Reff .* exp.(-c ./ mParams.Nh)
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

    nxGrowth = rand.(Poisson.(viDist.Reff .* viDist.nx .* simSet.dt))
    nxGrowth = clamp.(nxGrowth, zero(nxGrowth), viDist.nx)
    viDist.nx += nxGrowth
end

"""
   mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer})

Mutates the viral distribution in `viDist` with the mutations given by `nxMutated`, following `mParams.mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer})

    viralJumps = zero(nxMutated)

    idxGrowth = findall(x -> x > 0, nxMutated)
    for idx in idxGrowth
        jump = mutateNVirusAt!(viDist, mParams.mutationKernel, nxMutated[idx], idx)
        viralJumps += jump
    end

    viDist.nx += viralJumps - nxMutated
end

"""
   mutateNVirusAt!(viDist::viralImmuneDistribution, mParams::modelParams, N::Integer, idx::Integer)::Vector{<:Integer}

Mutates `N` virus from position `idx` to positions sampled from `mParams.mutationKernel`, returning a vector with the mutated viral density.

# Examples
```julia-repl
julia>
```
"""
function mutateNVirusAt!(viDist::viralImmuneDistribution, mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer, idx::Integer)::Vector{<:Integer}

    numVirus = viDist.nx[idx]

    numVirus == 0 && throw(ArgumentError("There is no virus at index $idx"))

    N > 0 || throw(ArgumentError("The number of viruses to mutate must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to mutate ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    mutDisplacements = getMutationDisplacements(mutationKernel, N)
    newIndices = round.(Int, mutDisplacements .+ idx)
    clamp!(newIndices, 1, length(viDist.space))
    
    mutJumps = zero(viDist.nx)
    [mutJumps[idx] += 1 for idx in newIndices]

    return mutJumps
end

"""
   getMutationDisplacements(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)Vector{<:Real}

Returns `N` displacements sampled from distibution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)::Vector{<:Real}
    return Distributions.rand!(mutationKernel, zeros(N))
end

"""
    mutationStep!(viDist::viralImmuneDistribution, mParams::modelParams,simSet::simulationConfig, t::Real = 0)::Vector{<:Integer}

Performs a mutation simulation step in the viral distribution `viDist`, with homogenous rate `mParams.mutationRate`, jumps sampled from `mParams.mutationKernel` and timestep `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function mutationStep!(viDist::viralImmuneDistribution, mParams::modelParams,simSet::simulationConfig, t::Real = 0)::Vector{<:Integer}

    nxMutated = rand.(Poisson.(mParams.mutationRate .* viDist.nx .* simSet.dt))
    nxMutated = clamp.(nxMutated, zero(nxMutated), viDist.nx)
    
    mutateViralDistribution!(viDist, mParams, nxMutated)
    return nxMutated
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

    nxDeath = rand.(Poisson.(viDist.nx .* simSet.dt))
    nxDeath = clamp.(nxDeath, zero(nxDeath), viDist.nx)

    viDist.nx -= nxDeath
    return nxDeath
end

"""
   simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Bool

Performs a simulation step in the viral distribution `viDist`, with parameters `mParams`, timestep `simSet.dt`, annotates it at time `t`, and returns the extinction state (true if not extinct) and whether a non-local event has happened or not in the last step.

# Examples
```julia-repl
julia>
```
"""
function simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig)::Bool

    reproductionStep!(viDist, simSet)
    mutationStep!(viDist, mParams, simSet)
    hxGrowth = deathStep!(viDist, simSet)
    getImmuneUpdate!(viDist, mParams, hxGrowth)

    return iszero(viDist.nx)
end

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
function sampleDistributions!(viDist::viralImmuneDistribution, sampDists::sampledResults)
    push!(sampDists.nxt, viDist.nx)
    push!(sampDists.hxt, viDist.hx)
    push!(sampDists.Refft, viDist.Reff)
end

function plotConfig()
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
            linewidth=1, framestyle=:box, label=nothing, grid=false)
    gr()
end

function animateDistributions(mParams::modelParams, sampDists::sampledResults)
    
    animation = @animate for i in 1:length(sampDists.nxt)
        p = plot(sampDists.space, sampDists.nxt[i], colour=:coral, ylims=[0, maximum(maximum.(sampDists.nxt))], ylabel=raw"Viral density", xlabel=raw"$x$", label = "")
        plot!(twinx(), sampDists.space, sampDists.hxt[i] ./ mParams.Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"Immune memories", ylims = [0, 1], label = "")
        plot!([], [], color = :coral, label = raw"$n(x,t)$")
        plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
    end

    g = gif(animation)
    display(g)
    return g
end

# end