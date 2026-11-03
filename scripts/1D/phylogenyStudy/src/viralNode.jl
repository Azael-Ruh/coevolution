using StatsBase

mutable struct viralNode
    birthTime::Real
    position::Real
    children::Vector{viralNode}
    parent::Union{viralNode,Nothing}
    isSampled::Bool
    divergenceFromRoot::Int
    fitness::Real
end

import Base: broadcastable, show

function show(io::IO, vNode::viralNode)
    print(io, "viralNode with t = ", vNode.birthTime, ",  x = ", vNode.position, ", ", "children = ", length(vNode.children), ", ", isnothing(vNode.parent) ? "no" : "a", " parent, ", vNode.isSampled ? "isSampled = true, " : "isSampled = false, ", "divergence = ", vNode.divergenceFromRoot, ", fitness = ", vNode.fitness, ".")
end

# For broadcastability
function broadcastable(vNode::viralNode)
    return Ref(vNode)
end

"""
    viralNode(t::Real = 0, pos::Real = 0)::viralNode

Produce an empty viral node born at time `t` at position `pos` without parent.

# Examples
```julia-repl
julia>
```
"""
function viralNode(t::Real=0, pos::Real=0)::viralNode
    viralNode(t, pos, [], nothing, false, 0, 0)
end

"""
    viralNode()::viralNode

Produce an empty viral node born at time `t` at position `pos` from parent `pNode`.

# Examples
```julia-repl
julia>
```
"""
function viralNode(pNode::viralNode, t::Real=0, pos::Real=0, fitness::Real=0)
    cNode = viralNode(t, pos, [], pNode, false, pNode.divergenceFromRoot, fitness)
    push!(pNode.children, cNode)
    cNode
end

getBirthTime(node::viralNode) = node.birthTime
getDivergence(node::viralNode) = node.divergenceFromRoot
getFitness(node::viralNode) = node.fitness

"""
    reproduceNode!(node::viralNode, t::Real)::Vector{viralNode}

Reproduce the viral node `node` and anotate it with time `t` and returns the two children nodes.

# Examples
```julia-repl
julia>
```
"""
function reproduceNode!(node::viralNode, t::Real, fitness::Real=0)::Vector{viralNode}

    if !isempty(node.children)
        throw(ArgumentError("The given node is not a leaf!"))
    end

    nodePos = node.position
    viralNode(node, t, nodePos, fitness)
    viralNode(node, t, nodePos, fitness)

    return node.children
end

"""
    mutateNode(vTree::viralTree, idx::Int, newPosition::Real, t::Real)::viralNode

Mutate the the viral node `node` to position `newPosition` at time `t`.

# Examples
```julia-repl
julia>
```
"""
function mutateNode!(node::viralNode, newPosition::Real, t::Real, newfitness::Real=0)::viralNode
    node.position = newPosition
    node.birthTime = t
    node.divergenceFromRoot += 1
    node.fitness = newfitness
    return node
end

"""
    sampleNode(node::viralNode)

Mark `node` as sampled
"""
function sampleNode!(node::viralNode)
    node.isSampled = true
end

isSampled(node::viralNode) = node.isSampled

"""
    killNode(vTree::viralTree, idx::Int)

Mark the viral node `node` as dead and prune the tree as needed.

# Examples
```julia-repl
julia>
```
"""
function killNode!(node::viralNode)

    if !isempty(node.children)
        throw(ArgumentError("The given node is not a leaf!"))
    end

    if isnothing(node.parent)
        throw(ArgumentError("The given node is the anchor node of the tree"))
    end

    isSampled(node) && return

    pNode = node.parent
    pNode.children = filter(x -> x !== node, pNode.children)

    while isempty(pNode.children)
        isSampled(pNode) && return
        node = pNode
        pNode = node.parent
        pNode.children = filter(x -> x !== node, pNode.children)
    end

    while isnothing(pNode.parent) && (length(pNode.children) == 1)
        isSampled(pNode) && return
        pNode = first(pNode.children)
        pNode.parent = nothing
    end

end

"""
    getTMCA(node1::viralNode, node2::viralNode)

Obtain the time to the most recent common ancestor (MRCA) of nodes `node1` and `node2`.

# Examples
```julia-repl
julia>
```
"""
function getMRCAtime(node1::viralNode, node2::viralNode)::Real

    tPresent = max(node1.birthTime, node2.birthTime)

    #TODO: think about this when all nodes have same times!
    while !(node1 == node2)
        if node2.birthTime >= node1.birthTime
            isnothing(node2.parent) ? (return Inf) : (node2 = node2.parent)
        else
            isnothing(node1.parent) ? (return Inf) : (node1 = node1.parent)
        end
    end

    tMRCA = tPresent - node1.birthTime
    return tMRCA
end

"""
    getMRCAtimes(allLeaves::Vector{viralNode}, NSampling::Integer=0)::Vector{<:Real}

Get an `NSampling` leaves based random sample (or full if NSampling = 0) of the two point times to the Most Recent Common Ancestor (MRCA) for viralNodes in `allLeaves`.  

# Examples
```julia-repl
julia>
```
"""
function getMRCAtimes(allLeaves::Vector{viralNode}, NSampling::Integer=0)::Vector{<:Real}

    Ntotal = length(allLeaves)

    if Ntotal <= 1
        Ntotal == 0 ? throw(ArgumentError("There is no virus in the given population")) : throw(ArgumentError("There is only one virus in the given population"))
    end

    NSampling < 0 && throw(ArgumentError("NSampling needs to be non-negative"))
    NSampling > Ntotal && throw(ArgumentError("NSampling needs to be smaller that the total number of leaves $Ntotal"))
    NSampling == Ntotal && (NSampling = 0)

    NSampling > 0 ? (selectedLeaves = sample(allLeaves, NSampling, replace=false)) : (selectedLeaves = allLeaves)

    MRCAtimes = Vector{Real}(undef, 0)
    for i in eachindex(selectedLeaves), j in eachindex(selectedLeaves)[1:(i-1)]
        push!(MRCAtimes, getMRCAtime(selectedLeaves[i], selectedLeaves[j]))
    end

    return MRCAtimes

end

function getSampledFromNode(node::viralNode)

    sampledNodes = viralNode[]
    nodes2check = [node]

    while !isempty(nodes2check)
        node = popfirst!(nodes2check)
        isSampled(node) && push!(sampledNodes, node)
        append!(nodes2check, node.children)
    end

    return sampledNodes
end

function getSampledTreeFromNode(node::viralNode, parentNode::Union{viralNode,Nothing})
    newAnchorNode = viralNode(node.birthTime, node.position, viralNode[], parentNode, node.isSampled, node.divergenceFromRoot, node.fitness)

    nodes2check = viralNode[]
    append!(nodes2check, node.children)
    childrenList = viralNode[]

    while !isempty(nodes2check)
        child = popfirst!(nodes2check)
        if isSampled(child)
            push!(childrenList, getSampledTreeFromNode(child, newAnchorNode))
            println("Found sampled node with $(length(child.children)) children")
        else
            append!(nodes2check, child.children)
        end
    end

    newAnchorNode.children = childrenList
    return newAnchorNode
end

## ======================================================================
#                           Plotting tools
#  ======================================================================

using BasicTreePlots, AbstractTrees, CairoMakie

import AbstractTrees: children
import BasicTreePlots: distance, label

function children(node::viralNode)::Vector{viralNode}
    return node.children
end

# Need to think about where to plot the split!!!
function distance(node::viralNode)::Real
    return node.birthTime - node.parent.birthTime
end

function label(node::viralNode)::String
    return "$(round(Int, node.divergenceFromRoot))"
end


"""
    plotTreeFromNode(node::viralNode)

Produce a tree plot for the biggest tree `node` is in using BasicTreePlots.

# Examples
```julia-repl
julia>
```
"""
function plotTreeFromNode(node::viralNode, depth::Real=-1; branchcolor=:black)

    while !isnothing(node.parent) && depth !== 0
        node = node.parent
        depth -= 1
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    tp = treeplot!(node, branchcolor=branchcolor)
    display(fig)

    return fig, ax, tp
end

nleaves(n) = isempty(n.children) ? 1 : sum(nleaves, n.children)

function ladderize!(n; rev=false)
    foreach(c -> ladderize!(c; rev), n.children)
    sort!(n.children, by=nleaves, rev=rev)
    return n
end


## =======================================
#       Tree dissimilarity indexes
#  =======================================

choose(N, n) = factorial(N) / (factorial(N-n) * factorial(n))

colessIndexBinaryTree(node::viralNode) = isempty(node.children) ? 0 : abs(nleaves(first(node.children)) - nleaves(last(node.children))) + sum(colessIndexBinaryTree, node.children)

normalisedColessBinaryIndex(node::viralNode) = colessIndexBinaryTree(node) / choose(nleaves(node) - 1, 2)

degree(node::viralNode) = length(node.children)
fFunction(n::Int) = log(n + ℯ)
deltaf(node::viralNode) = isempty(node.children) ? fFunction(0) : fFunction(degree(node)) + sum(deltaf, node.children)
MDM(x::Vector{<:Real}) = mean(abs.(x .- median(x)))

CollessLikeLogMDM(node::viralNode) = isempty(node.children) ? 0 : MDM(deltaf.(node.children)) + sum(CollessLikeLogMDM, node.children)
normalisedCollessLikeLogMDM(node::viralNode) = CollessLikeLogMDM(node) / ((fFunction(0) + fFunction(2)) * (nleaves(node)-1) * (nleaves(node)-2) / 4)
