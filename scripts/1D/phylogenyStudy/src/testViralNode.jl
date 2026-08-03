using Test

include("viralNode.jl")

@testset  "viralNode" begin

    @testset "constructor" begin
        node = viralNode()
        @test typeof(node) == viralNode
        @test node.children == []
        @test node.birthTime == 0
        @test node.position == 0
        @test isnothing(node.parent)

        nodeChild = viralNode(node)
        @test typeof(nodeChild) == viralNode
        @test nodeChild.children == []
        @test nodeChild.birthTime == 0
        @test nodeChild.position == 0
        @test nodeChild.parent == node
    end

    @testset "reproduceNode" begin
        node = viralNode()
        t = 1
        children = reproduceNode!(node, t)
        
        @test length(node.children) == 2
        @test first(node.children).birthTime == t
        @test last(node.children).birthTime == t
        @test last(node.children).children == []
        @test first(node.children).position == last(node.children).position
        @test first(node.children).position == node.position
        @test first(node.children).parent == node

        @test_throws ArgumentError reproduceNode!(node, t)

        @test children == node.children
    end

    @testset "mutateNode" begin
        t = 1
        node = viralNode()
        newPos = 1
        mutateNode!(node, newPos, t)

        @test node.children == []
        @test node.birthTime == t
        @test node.position == newPos
        @test isnothing(node.parent)
    end

    @testset "killNode" begin
        node = viralNode()

        @test_throws ArgumentError killNode!(node)

        t = 1
        reproduceNode!(node, t)

        @test_throws ArgumentError killNode!(node)

        child1 = first(node.children)
        child2 = last(node.children)
        killNode!(child1)

        @test length(node.children) == 1
        @test first(node.children) !== child1
        @test first(node.children) == child2

        node = viralNode()
        t = 1
        reproduceNode!(node, t)
        child1 = first(node.children)
        child2 = last(node.children)

        t = 2
        reproduceNode!(child1, t)
        child11 = first(child1.children)
        child12 = last(child1.children)

        t = 3
        reproduceNode!(child11, t)
        child111 = first(child11.children)
        child112 = last(child11.children)

        killNode!(child111)
        killNode!(child112)

        @test length(child1.children) == 1
        @test first(child1.children) !== child11
        @test first(child1.children) == child12

        killNode!(child12)
        @test child2.parent == nothing

        node = viralNode()
        t = 1
        reproduceNode!(node, t)
        child1 = first(node.children)
        child2 = last(node.children)

        t = 2
        reproduceNode!(child1, t)
        child11 = first(child1.children)
        child12 = last(child1.children)
        reproduceNode!(child2, t)
        child21 = first(child2.children)
        child22 = last(child2.children)

        killNode!(child12)
        killNode!(child21)
        killNode!(child11)

        @test child22.parent == nothing
    end

    @testset "getMRCAtime" begin
        node = viralNode()

        t = 1
        reproduceNode!(node, t)
        child1 = first(node.children)
        child2 = last(node.children)

        @test_throws ArgumentError getMRCAtime(child1, node)
        @test_throws ArgumentError getMRCAtime(node, child1)
        
        @test getMRCAtime(child1, child2) == 1
        
        t = 2
        reproduceNode!(child1, t)

        child11 = first(child1.children)
        child12 = last(child1.children)

        @test getMRCAtime(child11, child12) == 1
        @test getMRCAtime(child11, child11) == 0
        @test getMRCAtime(child11, child2) == 2

        node2 = viralNode()
        node3 = viralNode()
        
        @test getMRCAtime(node2, node3) == Inf

        t = 1
        children2 = reproduceNode!(node2, t)

        @test getMRCAtime(child11, children2[1]) == Inf

    end

end