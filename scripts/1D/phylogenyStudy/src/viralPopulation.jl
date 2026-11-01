include("viralNode.jl")

using StatsBase

mutable struct viralPopulation
    space::UnitRange{<:Integer}
    viralNodes::Vector{Vector{viralNode}}
end

"""
    viralPopulation(x::Vector{<:Real})::viralPopulation

Produce an empty viral population with space vector `x`.

# Examples
```julia-repl
julia>
```
"""
function viralPopulation(x::UnitRange{<:Integer})::viralPopulation

    viralNodes = [Vector{viralNode}(undef, 0) for i in x]

    viralPopulation(x, viralNodes)
end

"""
    initialiseViralPopulation(x::Vector{<:Real}, nx::Vector{<:Integer} = Vector{Int64}(undef, 0))::viralPopulation

Initialise a viral population with space vector `x` and viruses distributed as in `nx`.

# Examples
```julia-repl
julia>
```
"""
function initialiseViralPopulation(x::UnitRange{<:Integer}, nx::Vector{<:Integer}=Vector{Int64}(undef, 0))::viralPopulation

    isempty(nx) && return viralPopulation(x)
    nxAux = copy(nx)

    idxMax = length(x)
    viralNodes = [Vector{viralNode}(undef, 0) for i in 1:idxMax]

    while (idx = findfirst(nxAux .> 0)) !== nothing
        while nxAux[idx] > 0
            push!(viralNodes[idx], viralNode(0, x[idx], [], nothing, false, 0, 0))
            nxAux[idx] -= 1
        end
    end

    viralPopulation(x, viralNodes)
end

"""
    reproduceVirus!(vPop::viralPopulation, positionIdx::Integer, t::Real = 0)

Reproduce a random virus at position given by the index `positionIdx` in populaiton `vPop` at time `t`.

# Examples
```julia-repl
julia>
```
"""
function reproduceVirus!(vPop::viralPopulation, positionIdx::Integer, t::Real=0, fitness::Real=0)

    numVirus = length(vPop.viralNodes[positionIdx])

    numVirus == 0 && throw(ArgumentError("There is no virus at index $positionIdx"))

    randomIdx = rand(1:numVirus)
    append!(vPop.viralNodes[positionIdx], reproduceNode!(vPop.viralNodes[positionIdx][randomIdx], t, fitness))
    deleteat!(vPop.viralNodes[positionIdx], randomIdx)
end

"""
    reproduceNVirus!(vPop::viralPopulation, positionIdx::Integer, N::Integer, t::Real = 0)

Reproduce `N` random viruses at position given by the index `positionIdx` in populaiton `vPop` at time `t`.

# Examples
```julia-repl
julia>
```
"""
function reproduceNVirus!(vPop::viralPopulation, positionIdx::Integer, N::Integer, t::Real=0, fitness::Real=0)

    numVirus = length(vPop.viralNodes[positionIdx])

    numVirus == 0 && throw(ArgumentError("There is no virus at index $positionIdx"))

    N > 0 || throw(ArgumentError("The number of viruses to reproduce must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to reproduce ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    selectedIdx = sample(1:numVirus, N, replace=false)
    for idx in selectedIdx
        append!(vPop.viralNodes[positionIdx], reproduceNode!(vPop.viralNodes[positionIdx][idx], t, fitness))
        deleteat!(vPop.viralNodes[positionIdx], idx)
    end
end

"""
    mutateVirus!(vPop::viralPopulation, originalPositionIdx::Integer, newPositionIdx::Integer, t::Real = 0)

Mutate a random virus from `originalPositionIdx` to `newPositionIdx` in populaiton `vPop` at time `t`.

# Examples
```julia-repl
julia>
```
"""
function mutateVirus!(vPop::viralPopulation, originalPositionIdx::Integer, newPositionIdx::Integer, t::Real=0, newFitness::Real=0)

    numVirus = length(vPop.viralNodes[originalPositionIdx])

    numVirus == 0 && throw(ArgumentError("There is no virus at index $originalPositionIdx"))

    randomIdx = rand(1:numVirus)
    push!(vPop.viralNodes[newPositionIdx], mutateNode!(vPop.viralNodes[originalPositionIdx][randomIdx], vPop.space[newPositionIdx], t, newFitness))
    deleteat!(vPop.viralNodes[originalPositionIdx], randomIdx)
end

"""
    killVirus!(vPop::viralPopulation, positionIdx::Integer)

Kill a random virus at `positionIdx` in populaiton `vPop`.

# Examples
```julia-repl
julia>
```
"""
function killVirus!(vPop::viralPopulation, positionIdx::Integer)

    numVirus = length(vPop.viralNodes[positionIdx])

    numVirus == 0 && throw(ArgumentError("There is no virus at index $positionIdx"))

    randomIdx = rand(1:numVirus)
    try
        killNode!(vPop.viralNodes[positionIdx][randomIdx])
    catch e
        e.msg != "The given node is the anchor node of the tree" && throw(e)
    end

    deleteat!(vPop.viralNodes[positionIdx], randomIdx)
end

"""
    killNvirus!(vPop::viralPopulation, positionIdx::Integer, N::Integer)

Kill `N` random viruses at position given by the index `positionIdx` in populaiton `vPop`.

# Examples
```julia-repl
julia>
```
"""
function killNvirus!(vPop::viralPopulation, positionIdx::Integer, N::Integer)

    numVirus = length(vPop.viralNodes[positionIdx])

    numVirus == 0 && throw(ArgumentError("There is no virus at index $positionIdx"))

    N > 0 || throw(ArgumentError("The number of viruses to reproduce must be positive"))
    N > numVirus && throw(ArgumentError("The number of viruses to reproduce ($N) cannot exceed the number of viruses ($numVirus) in the designed position"))

    selectedIdx = sample(1:numVirus, N, replace=false)
    sort!(selectedIdx, rev=true)
    while !isempty(selectedIdx)

        idx = pop!(selectedIdx)

        try
            killNode!(vPop.viralNodes[positionIdx][idx])
        catch e
            e.msg != "The given node is the anchor node of the tree" && throw(e)
        end

        deleteat!(vPop.viralNodes[positionIdx], idx)
        selectedIdx .-= 1
    end
end

"""
    getMRCAtimes(vPop::viralPopulation, NSampling::Integer = 0)::Vector{<:Real}

Get an `NSampling` leaves based random sample (or full if NSampling = 0) of the two point times to the Most Recent Common Ancestor (MRCA) in `vPop`.  

# Examples
```julia-repl
julia>
```
"""
function getMRCAtimes(vPop::viralPopulation, NSampling::Integer=0)::Vector{<:Real}

    allLeaves = vcat(vPop.viralNodes...)
    return getMRCAtimes(allLeaves, NSampling)
end

function getViralDistribution(vPop::viralPopulation)
    return length.(vPop.viralNodes)
end

function getAnchorNode(vPop::viralPopulation)::viralNode
    node = first(vPop.viralNodes[findfirst(.!isempty.(vPop.viralNodes))])
    while (!isnothing(node.parent))
        node = node.parent
    end
    return node
end

function getAllSampledNodes(vPop::viralPopulation)::Vector{viralNode}
    anchorNode = getAnchorNode(vPop)
    return getSampledFromNode(anchorNode)
end

function getSampledTree(vPop::viralPopulation)::viralNode
    anchorNode = getAnchorNode(vPop)
    return getSampledTreeFromNode(anchorNode, nothing)
end

"""
    getFullSampledTree!(vPop::viralPopulation)::viralNode

Produces a viral tree from vPop.viralNodes that only keeps the branches that have sampled nodes. WARNING: kills the viral population, rendering vPop unusable.
"""
function getFullSampledTree!(vPop::viralPopulation)::viralNode
    anchorNode = getAnchorNode(vPop)
    for (idx, nVirus) in enumerate(getViralDistribution(vPop))
        nVirus > 0 && killNvirus!(vPop, idx, nVirus)
    end
    return anchorNode
end

"""
    getFullSampledTree!(vPop::viralPopulation)::viralNode

Produces a viral tree from vPop.viralNodes that only keeps the branches that have sampled nodes. WARNING: uses a deep copy of the population, it can double the memory use.
"""
function getFullSampledTreeCopy(vPop::viralPopulation)::viralNode
    vPop2kill = deepcopy(vPop)
    newAnchorNode = getAnchorNode(vPop2kill)
    for (idx, nVirus) in enumerate(getViralDistribution(vPop2kill))
        nVirus > 0 && killNvirus!(vPop2kill, idx, nVirus)
    end
    return newAnchorNode
end