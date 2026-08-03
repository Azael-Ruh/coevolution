mutable struct viralNode
    birthTime::Real
    position::Real
    children::Vector{viralNode}
    parent::Union{viralNode, Nothing}
end

import Base: broadcastable

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
function viralNode(t::Real = 0, pos::Real = 0)::viralNode
    viralNode(t, pos, [], nothing)
end

"""
    viralNode()::viralNode

Produce an empty viral node born at time `t` at position `pos` from parent `pNode`.

# Examples
```julia-repl
julia>
```
"""
function viralNode(pNode::viralNode, t::Real = 0, pos::Real = 0)
    viralNode(t, pos, [], pNode)
end

"""
    reproduceNode!(node::viralNode, t::Real)::Vector{viralNode}

Reproduce the viral node `node` and anotate it with time `t` and returns the two children nodes.

# Examples
```julia-repl
julia>
```
"""
function reproduceNode!(node::viralNode, t::Real)::Vector{viralNode}

    if !isempty(node.children)
        throw(ArgumentError("The given node is not a leaf!"))
    end

    nodePos = node.position
    child1 = viralNode(node, t, nodePos)
    child2 = viralNode(node, t, nodePos)

    return node.children = [child1, child2]
end

"""
    mutateNode(vTree::viralTree, idx::Int, newPosition::Real, t::Real)::viralNode

Mutate the the viral node `node` to position `newPosition` at time `t`.

# Examples
```julia-repl
julia>
```
"""
function mutateNode!(node::viralNode, newPosition::Real, t::Real)::viralNode
    node.position = newPosition
    node.birthTime = t
    return node
end

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

    pNode = node.parent
    pNode.children = filter(x -> x !==node, pNode.children)

    while isempty(pNode.children)
        node = pNode
        pNode = node.parent
        pNode.children = filter(x -> x !==node, pNode.children)
    end

    while (pNode.parent == nothing) && (length(pNode.children) == 1)
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

    if !isempty(node1.children) || !isempty(node2.children)
        throw(ArgumentError("One of the given nodes is not a leaf!"))
    end

    tPresent = max(node1.birthTime, node2.birthTime)

    #TODO: think about this when all nodes have same times!
    while !(node1 == node2)    
        if node2.birthTime >= node1.birthTime
            node2.parent == nothing ? (return Inf) : (node2 = node2.parent)
        else
            node1.parent == nothing ? (return Inf) : (node1 = node1.parent)
        end
    end

    tMRCA = tPresent - node1.birthTime
    return tMRCA
end

# =======================================================================
#                           Plotting tools
# =======================================================================

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
    return "x=$(round(Int, node.position))"
end


"""
    plotTreeFromNode(node::viralNode)

Produce a tree plot for the biggest tree `node` is in using BasicTreePlots.

# Examples
```julia-repl
julia>
```
"""
function plotTreeFromNode(node::viralNode, depth::Real = -1)
    
    while node.parent != nothing && depth != 0
        node = node.parent
        depth -= 1
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    treeplot!(node, tipfontsize = 14)
    display(fig)

    return fig
end