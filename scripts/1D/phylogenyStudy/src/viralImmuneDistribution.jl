include("viralPopulation.jl")
include(expanduser("../../../../code/simulate/coevolution1DSimulationTools.jl"))

struct modelParams
    r::Real
    R0::Real
    s::Real
    Nh::Real
    mutationRate::Real
    mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}, piecewiseKernel}
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
    nx::Vector{Integer}
    hx::Vector{Integer}
    viralPop::viralPopulation
    Reff::Vector{<:Real}
end

"""
    viralImmuneDistribution(x::Vector{<:Real}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution

Produces an instance of viralImmuneDistribution population with space vector `x`, viral distribution `nx`, immune distribution `hx` and viral population initialised from `nx`.

# Examples
```julia-repl
julia>
```
"""
function viralImmuneDistribution(x::UnitRange{<:Integer}, nx::Vector{<:Integer}, hx::Vector{<:Integer})::viralImmuneDistribution
    return viralImmuneDistribution(x, nx, hx, initialiseViralPopulation(x, nx), ones(Float64, length(x)))
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
    
    c = conv(viDist.hx, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
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
   reproduceViralDistribution!(viDist::viralImmuneDistribution, nxGrowth::Vector{<:Integer})

Reproduces the viral distribution in `viralImmuneDistribution` with the births given by `nxGrowth`.

# Examples
```julia-repl
julia>
```
"""
function reproduceViralDistribution!(viDist::viralImmuneDistribution, nxGrowth::Vector{<:Integer}, t::Real = 0)

    viDist.nx += nxGrowth

    idxGrowth = findall(x -> x > 0, nxGrowth)
    for idx in idxGrowth
        reproduceNVirus!(viDist.viralPop, idx, nxGrowth[idx], t)
    end
end

"""
   reproductionStep!(viDist::viralImmuneDistribution, simSet::simulationConfig)::Vector{<:Integer}

Performs a reproduction simulation step in the viral distribution `viDist`, following `viDist.Reff` and the timestep in `simSet.dt`.

# Examples
```julia-repl
julia>
```
"""
function reproductionStep!(viDist::viralImmuneDistribution, simSet::simulationConfig, t::Real = 0)::Vector{<:Integer}

    nxGrowth = rand.(Poisson.(viDist.Reff .* viDist.nx .* simSet.dt))
    nxGrowth = clamp.(nxGrowth, zero(nxGrowth), viDist.nx)

    reproduceViralDistribution!(viDist, nxGrowth, t)
    return nxGrowth
end

"""
   mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer}, t::Real = 0)

Mutates the viral distribution in `viDist` with the mutations given by `nxMutated`, following `mParams.mutationKernel` and annotating it at time `t`.

# Examples
```julia-repl
julia>
```
"""
function mutateViralDistribution!(viDist::viralImmuneDistribution, mParams::modelParams, nxMutated::Vector{<:Integer}, t::Real = 0)

    viralJumps = zero(nxMutated)

    idxGrowth = findall(x -> x > 0, nxMutated)
    for idx in idxGrowth
        viralJumps += mutateNVirusAt!(viDist, mParams.mutationKernel, nxMutated[idx], idx, t)
    end

    viDist.nx += viralJumps - nxMutated
end

"""
   mutateNVirusAt!(viDist::viralImmuneDistribution, mParams::modelParams, N::Integer, idx::Integer, t::Real = 0)::Vector{<:Integer}

Mutates `N` virus from position `idx` to positions sampled from `mParams.mutationKernel` and annotates it at time `t` in `viDist`, returning a vector with the mutated viral density. CAUTION: does not update `viDist.nx`

# Examples
```julia-repl
julia>
```
"""
function mutateNVirusAt!(viDist::viralImmuneDistribution, mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}, piecewiseKernel}, N::Integer, idx::Integer, t::Real = 0)::Vector{<:Integer}

    numVirus = viDist.nx[idx]

    numVirus == 0 && throw(ArgumentError("There is no virus at index $idx"))

    N > 0 || throw(ArgumentError("The number of viruses to mutate must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to mutate ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    #TODO: think about non-local jump flagging
    mutDisplacements = getMutationDisplacements(mutationKernel, N)
    newIndices = round.(Int, mutDisplacements .+ idx)
    clamp!(newIndices, 1, length(viDist.space))

    for newIdx in newIndices
        mutateVirus!(viDist.viralPop, idx, newIdx, t)
    end

    mutJumps = zero(viDist.nx)
    [mutJumps[idx] += 1 for idx in newIndices]

    return mutJumps
end

"""
   getMutationDisplacements(mutationKernel::Union{Distribution{Univariate, Continuous}, Distribution{Univariate, Discrete}}, N::Integer)::Vector{<:Real}

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
   getMutationDisplacements(mutationKernel::piecewiseKernel, Distribution{Univariate, Discrete}}, N::Integer)::Vector{<:Real}

Returns `N` displacements sampled from the picewise distribution `mutationKernel`.

# Examples
```julia-repl
julia>
```
"""
function getMutationDisplacements(mutationKernel::piecewiseKernel, N::Integer)::Vector{<:Real}
    nNonLocal = rand(Binomial(N, mutationKernel.nonLocalMutProb))
    return [Distributions.rand!(localKernel, zeros(N - nNonLocal)); mutationKernel.nonLocalJump .* ones(nNonLocal)]
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
    
    mutateViralDistribution!(viDist, mParams, nxMutated, t)
    return nxMutated
end


"""
   killViralDistribution!(viDist::viralImmuneDistribution, nxDeath::Vector{<:Integer})

Kills the viral distribution in `viDist` as given by `nxDeath`.

# Examples
```julia-repl
julia>
```
"""
function killViralDistribution!(viDist::viralImmuneDistribution, nxDeath::Vector{<:Integer})
    viDist.nx -= nxDeath

    idxDeath = findall(x -> x > 0, nxDeath)
    for idx in idxDeath
        killNvirus!(viDist.viralPop, idx, nxDeath[idx])
    end
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

    killViralDistribution!(viDist, nxDeath)
    return nxDeath
end

"""
   simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig, t::Real = 0)::Bool

Performs a simulation step in the viral distribution `viDist`, with parameters `mParams`, timestep `simSet.dt`, annotates it at time `t`, and returns the extinction state (true if not extinct) of the distribution.

# Examples
```julia-repl
julia>
```
"""
function simulationStep!(viDist::viralImmuneDistribution, mParams::modelParams, simSet::simulationConfig, t::Real = 0)::Bool

    reproductionStep!(viDist, simSet, t)
    mutationStep!(viDist, mParams, simSet, t)
    hxGrowth = deathStep!(viDist, simSet)
    getImmuneUpdate!(viDist, mParams, hxGrowth)

    return !iszero(viDist.nx)
end

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
    newIdx = idxWave .+ firstIdx .- first(idxWave)

    newNx = zero(viDist.nx)
    newNx[newIdx] = viDist.nx[idxWave]

    newHx = zero(viDist.hx)
    newHx[1:last(newIdx)] = viDist.hx[last(idxWave) - last(newIdx) + 1 : last(idxWave)]

    newViralNodes = [Vector{viralNode}(undef, 0) for i in eachindex(newNx)]
    newViralNodes[newIdx] = viDist.viralPop.viralNodes[idxWave]

    viDist.nx = newNx
    viDist.hx = newHx
    viDist.viralPop.viralNodes = newViralNodes

    getGrowthRate!(viDist, mParams)
end