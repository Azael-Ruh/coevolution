using Test

include("viralPopulation.jl")

@testset  "viralPopulation" begin

    @testset "constructors" begin
        x = 0:10
        population = viralPopulation(x)

        @test typeof(population) == viralPopulation
        @test typeof(population.viralNodes) == Vector{Vector{viralNode}}
        @test population.space == x
        @test length(population.viralNodes) == length(x)

        x = 0:10
        population = initialiseViralPopulation(x)

        @test typeof(population) == viralPopulation
        @test typeof(population.viralNodes) == Vector{Vector{viralNode}}
        @test population.space == x
        @test length(population.viralNodes) == length(x)

        idxVirus = 6
        nx = zero(x)
        nx[idxVirus] = 1
        population = initialiseViralPopulation(x, nx)

        @test typeof(population) == viralPopulation
        @test typeof(population.viralNodes) == Vector{Vector{viralNode}}
        @test population.space == x
        @test length(population.viralNodes) == length(x)
        @test length.(population.viralNodes) == nx
        @test typeof(first(population.viralNodes[idxVirus])) == viralNode

        idxVirus = [5, 6, 7]
        nx = zero(x)
        nx[idxVirus] = [1, 3, 1]
        population = initialiseViralPopulation(x, nx)

        @test length(population.viralNodes) == length(x)
        @test length.(population.viralNodes) == nx
        @test typeof(first(population.viralNodes[idxVirus[1]])) == viralNode
        @test typeof(first(population.viralNodes[idxVirus[2]])) == viralNode
        @test typeof(last(population.viralNodes[idxVirus[2]])) == viralNode
        @test first(population.viralNodes[idxVirus[2]]) != last(population.viralNodes[idxVirus[2]])
        @test first(population.viralNodes[idxVirus[2]]).parent == nothing
        @test first(population.viralNodes[idxVirus[2]]).children == []
        @test first(population.viralNodes[idxVirus[2]]).birthTime == 0
    end

    @testset "reproduceVirus" begin

        x = 0:10
        population = initialiseViralPopulation(x)
        idxPos = 6

        @test_throws ArgumentError reproduceVirus!(population, idxPos)

        nx = zero(x)
        nx[idxPos] = 1
        population = initialiseViralPopulation(x, nx)
        t = 1
        reproduceVirus!(population, idxPos, t)

        @test sum(length.(population.viralNodes)) == sum(nx) + 1
        @test length(population.viralNodes[idxPos]) == nx[idxPos] + 1
        @test first(population.viralNodes[idxPos]) != last(population.viralNodes[idxPos])  
        @test first(population.viralNodes[idxPos]).birthTime == t
        @test first(population.viralNodes[idxPos]).parent == last(population.viralNodes[idxPos]).parent

        nx = zero(x)
        nx[idxPos] = 3
        population = initialiseViralPopulation(x, nx)
        t = 1
        reproduceVirus!(population, idxPos, t)
        fatherNode = last(population.viralNodes[idxPos]).parent

        @test sum(length.(population.viralNodes)) == sum(nx) + 1
        @test length(population.viralNodes[idxPos]) == nx[idxPos] + 1
        @test first(population.viralNodes[idxPos]) != last(population.viralNodes[idxPos])  
        @test filter(x -> x != fatherNode, population.viralNodes[idxPos]) == population.viralNodes[idxPos]
        @test length(filter(x -> x.parent == fatherNode, population.viralNodes[idxPos])) == 2
        @test length(filter(x -> x.parent != fatherNode, population.viralNodes[idxPos])) == nx[idxPos] - 1
        @test filter(x -> x.parent != fatherNode, population.viralNodes[idxPos]) == population.viralNodes[idxPos][1:end-2]

        x = 0:10
        population = initialiseViralPopulation(x)
        idxPos = 6
        N = 1

        @test_throws ArgumentError reproduceNVirus!(population, idxPos, N)

        nx = zero(x)
        nx[idxPos] = 3
        population = initialiseViralPopulation(x, nx)
        t = 1
        N = 0

        @test_throws ArgumentError reproduceNVirus!(population, idxPos, N)

        N = 4

        @test_throws ArgumentError reproduceNVirus!(population, idxPos, N)

        N = 1
        reproduceNVirus!(population, idxPos, N)
        fatherNode = last(population.viralNodes[idxPos]).parent

        @test sum(length.(population.viralNodes)) == sum(nx) + 1
        @test length(population.viralNodes[idxPos]) == nx[idxPos] + 1
        @test first(population.viralNodes[idxPos]) != last(population.viralNodes[idxPos])  
        @test filter(x -> x != fatherNode, population.viralNodes[idxPos]) == population.viralNodes[idxPos]
        @test length(filter(x -> x.parent == fatherNode, population.viralNodes[idxPos])) == 2
        @test length(filter(x -> x.parent != fatherNode, population.viralNodes[idxPos])) == nx[idxPos] - 1
        @test filter(x -> x.parent != fatherNode, population.viralNodes[idxPos]) == population.viralNodes[idxPos][1:end-2]

        nx = zero(x)
        nx[idxPos] = 3
        population = initialiseViralPopulation(x, nx)
        t = 1
        N = 2

        reproduceNVirus!(population, idxPos, N)

        fatherNodes = [population.viralNodes[idxPos][i].parent for i = eachindex(population.viralNodes[idxPos])[end-3:2:end]]

        @test sum(length.(population.viralNodes)) == sum(nx) + 2
        @test length(population.viralNodes[idxPos]) == nx[idxPos] + 2
        @test filter(x -> any(fatherNodes .!= x), population.viralNodes[idxPos]) == population.viralNodes[idxPos]
        @test length(filter(x -> any(fatherNodes .== x.parent), population.viralNodes[idxPos])) == 4
        @test length(filter(x -> all(fatherNodes .!= x.parent), population.viralNodes[idxPos])) == nx[idxPos] - 2
    end

    @testset "mutateVirus" begin

        x = 0:10
        population = initialiseViralPopulation(x)
        idxPos = 6
        newIdxPos = 7

        @test_throws ArgumentError mutateVirus!(population, idxPos, newIdxPos)

        nx = zero(x)
        nx[idxPos] = 1
        population = initialiseViralPopulation(x, nx)
        t = 1

        ogVirus = first(population.viralNodes[idxPos])
        parent = ogVirus.parent
        children = ogVirus.children

        mutateVirus!(population, idxPos, newIdxPos, t)

        @test sum(length.(population.viralNodes)) == sum(nx)
        @test length(population.viralNodes[idxPos]) == nx[idxPos] - 1
        @test length(population.viralNodes[newIdxPos]) == nx[newIdxPos] + 1
        @test first(population.viralNodes[newIdxPos]).birthTime == t  
        @test first(population.viralNodes[newIdxPos]).parent == parent
        @test first(population.viralNodes[newIdxPos]).children == children

        idxVirus = [5, 6, 7]
        nx = zero(x)
        nx[idxVirus] = [1, 3, 1]
        population = initialiseViralPopulation(x, nx)

        idxPos = 6
        newIdxPos = 7

        mutateVirus!(population, idxPos, newIdxPos, t)

        ogVirus = last(population.viralNodes[newIdxPos])

        @test sum(length.(population.viralNodes)) == sum(nx)
        @test length(population.viralNodes[idxPos]) == nx[idxPos] - 1
        @test length(population.viralNodes[newIdxPos]) == nx[newIdxPos] + 1
        @test last(population.viralNodes[newIdxPos]).birthTime == t
        @test last(population.viralNodes[newIdxPos]).position == population.space[newIdxPos] 
        @test isempty(filter(x -> x == ogVirus, population.viralNodes[idxPos])) 
        @test length( filter(x -> x != ogVirus, population.viralNodes[idxPos])) == nx[idxPos] - 1
    end

    @testset "killVirus" begin
        
        x = 0:10
        population = initialiseViralPopulation(x)
        idxPos = 6

        @test_throws ArgumentError killVirus!(population, idxPos)

        nx = zero(x)
        nx[idxPos] = 1
        population = initialiseViralPopulation(x, nx)
        t = 1

        killVirus!(population, idxPos)

        @test all(isempty.(population.viralNodes))
        @test length(population.viralNodes[idxPos]) == nx[idxPos] - 1
        @test length.(population.viralNodes[1:idxPos-1]) == nx[1:idxPos-1]
        @test length.(population.viralNodes[idxPos+1:end]) == nx[idxPos+1:end]

        idxVirus = [5, 6, 7]
        nx = zero(x)
        nx[idxVirus] = [1, 3, 1]
        population = initialiseViralPopulation(x, nx)

        killVirus!(population, idxPos)

        @test sum(length.(population.viralNodes)) == sum(nx) - 1
        @test length(population.viralNodes[idxPos]) == nx[idxPos] - 1
        @test length.(population.viralNodes[1:idxPos-1]) == nx[1:idxPos-1]
        @test length.(population.viralNodes[idxPos+1:end]) == nx[idxPos+1:end]
        # @test isempty(filter(x -> x == killedVirus, population.viralNodes[idxPos])) 

        idxVirus = [5, 6, 7]
        nx = zero(x)
        nx[idxVirus] = [1, 1, 1]
        population = initialiseViralPopulation(x, nx)

        t = 1
        reproduceVirus!(population, idxPos, t)

        killVirus!(population, idxPos)

        @test sum(length.(population.viralNodes)) == sum(nx) + 1 - 1
        @test length(population.viralNodes[idxPos]) == nx[idxPos] + 1 - 1
        @test length.(population.viralNodes[1:idxPos-1]) == nx[1:idxPos-1]
        @test length.(population.viralNodes[idxPos+1:end]) == nx[idxPos+1:end]
        # @test isempty(filter(x -> x == killedVirus, population.viralNodes[idxPos])) 
        @test first(population.viralNodes[idxPos]).parent == nothing

        # More complex testing needs randomness injection!

        nx = zero(x)
        nx[idxPos] = 3
        population = initialiseViralPopulation(x, nx)
        
        t = 1
        N = 0
        @test_throws ArgumentError killNvirus!(population, idxPos, N)

        N = 4
        @test_throws ArgumentError killNvirus!(population, idxPos, N)

        N = 2
        killNvirus!(population, idxPos, N)

        @test sum(length.(population.viralNodes)) == sum(nx) - N
        @test length(population.viralNodes[idxPos]) == nx[idxPos] - N
        @test length.(population.viralNodes[1:idxPos-1]) == nx[1:idxPos-1]
        @test length.(population.viralNodes[idxPos+1:end]) == nx[idxPos+1:end]
    end

    @testset "getMRCAtimes" begin
        
        x = 0:10
        population = initialiseViralPopulation(x)

        @test_throws ArgumentError getMRCAtimes(population)

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 1
        population = initialiseViralPopulation(x, nx)

        @test_throws ArgumentError getMRCAtimes(population)

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 2
        population = initialiseViralPopulation(x, nx)

        @test getMRCAtimes(population) == [Inf]

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 3
        population = initialiseViralPopulation(x, nx)

        @test getMRCAtimes(population) == Inf .* ones(3)

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 1
        population = initialiseViralPopulation(x, nx)

        t = 1
        reproduceVirus!(population, posIdx, t)

        @test getMRCAtimes(population) == [1]

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 3
        population = initialiseViralPopulation(x, nx)

        @test_throws ArgumentError getMRCAtimes(population, -1)
        @test_throws ArgumentError getMRCAtimes(population, 4)
        @test getMRCAtimes(population, 2) == [Inf]
        @test getMRCAtimes(population, 3) == Inf .* ones(3)

        nx = zero(x)
        posIdx = 6
        nx[posIdx] = 3
        population = initialiseViralPopulation(x, nx)
        t = 1
        reproduceVirus!(population, posIdx, t)

        @test sort(getMRCAtimes(population)) == [1; Inf .* ones(5)]
    end

    @testset "getViralDistribution" begin

        x = 0:10
        population = initialiseViralPopulation(x)

        @test getViralDistribution(population) == zero(x)

        idxVirus = [5, 6, 7]
        nx = zero(x)
        nx[idxVirus] = [1, 3, 1]
        population = initialiseViralPopulation(x, nx)

        @test getViralDistribution(population) == nx

    end

end