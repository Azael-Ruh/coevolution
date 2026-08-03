using Test

include("viralImmuneDistribution.jl")

@testset  "viralImmuneDistribution" begin

    @testset "constructors" begin

        x = 0:10
        nx0 = zero(x)
        hx0 = zero(x)
            
        viDist = viralImmuneDistribution(x, nx0, hx0)

        @test viDist.space == x
        @test viDist.nx == nx0
        @test viDist.hx == hx0
        @test getViralDistribution(viDist.viralPop) == nx0
        @test viDist.Reff == ones(length(x))

        r = 0
        R0 = 1.2
        Nh::Int = 1e6
        mu = 0.2
        mutationKernel = Normal(0,1)
        mParams = modelParams(r, R0, Nh, mu, mutationKernel)

        @test mParams.s == (r > 0 ? log(R0)/r : 0)
        @test mParams.D == mu * std(mutationKernel)^2 / 2
        @test mParams.HkernelHalfLength ==  floor((10*ceil(r) + 1) / 2)

        getGrowthRate!(viDist, mParams)

        @test viDist.Reff == R0 .* ones(length(x))

        hGrowth = hx0
        idxGrowth = 6
        hGrowth[idxGrowth] = 1
        getGrowthRateUpdate!(viDist, mParams, hGrowth)

        @test viDist.Reff == begin
            c = conv(viDist.hx .+ hGrowth, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
            mParams.R0 .* exp.(-c ./ mParams.Nh)
        end

        viDist = viralImmuneDistribution(x, nx0, hGrowth)
        getGrowthRate!(viDist, mParams)
        getGrowthRateUpdate!(viDist, mParams, hGrowth)

        @test viDist.Reff == begin
            c = conv(viDist.hx .+ hGrowth, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
            mParams.R0 .* exp.(-c ./ mParams.Nh)
        end

        viDist = viralImmuneDistribution(x, nx0, hGrowth)
        getImmuneUpdate!(viDist, mParams, hGrowth)

        @test viDist.hx == 2 .* hGrowth

        tmax = 10
        dt = 0.1
        simSet = simulationConfig(tmax, dt)
        
        @test simSet.dtSampling == 1
        @test simSet.idxSampling == 10
    end

    @testset "reproduction" begin
        
        x = 0:10
        nx0 = zero(x)
        hx0 = zero(x)
        idxGrowth = 6
        nx0[idxGrowth] = 1
        viDist = viralImmuneDistribution(x, nx0, hx0)

        nxGrowth = nx0
        nxGrowth[idxGrowth] = 1

        reproduceViralDistribution!(viDist, nxGrowth)

        @test viDist.nx == nx0 .+ nxGrowth
        @test viDist.nx == getViralDistribution(viDist.viralPop)
        
    end

    @testset "mutation" begin

        x = 0:10
        nx0 = zero(x)
        hx0 = zero(x)
        idxMutate = 6
        nx0[idxMutate] = 1
        viDist = viralImmuneDistribution(x, nx0, hx0)

        r = 0
        R0 = 1.2
        Nh::Int = 1e6
        mu = 0.2
        mutationKernel = Normal(0,1)
        mParams = modelParams(r, R0, Nh, mu, mutationKernel)

        t = 1
        N = 0        
        @test_throws ArgumentError mutateNVirusAt!(viDist, mParams, N, idxMutate, t)

        N = 1
        @test_throws ArgumentError mutateNVirusAt!(viDist, mParams, N, idxMutate - 1, t)

        N = 2
        @test_throws ArgumentError mutateNVirusAt!(viDist, mParams, N, idxMutate, t)

        N = 1
        mutateNVirusAt!(viDist, mParams, N, idxMutate, t)
        viDist.nx = length.(viDist.viralPop.viralNodes)

        @test sum(viDist.nx) == sum(nx0)

        nx0[idxMutate] = 20
        viDist = viralImmuneDistribution(x, nx0, hx0)

        N = 10
        mutateNVirusAt!(viDist, mParams, N, idxMutate, t)
        viDist.nx = length.(viDist.viralPop.viralNodes)

        @test sum(viDist.nx) == sum(nx0)
        @test viDist.nx[idxMutate] < nx0[idxMutate]

        # Random tests waiting for Gustavo!

    end

    @testset "deaths" begin

        x = 0:10
        nx0 = zero(x)
        hx0 = zero(x)
        idxPos = 6
        nx0[idxPos] = 3
        viDist = viralImmuneDistribution(x, nx0, hx0)

        nxDeath = zero(nx0)
        nxDeath[idxPos] = 1
        killViralDistribution!(viDist, nxDeath)

        @test viDist.nx[idxPos] == nx0[idxPos] - nxDeath[idxPos]
        @test viDist.nx == length.(viDist.viralPop.viralNodes)

        # More tests wait for Gustavo
        
    end

    @testset "translate" begin

        x = 0:20
        nx0 = zero(x)
        hx0 = zero(x)
        idxPos = 16
        hx0[1:idxPos] .= 1
        nx0[idxPos] = 3
        viDist = viralImmuneDistribution(x, nx0, hx0)

        r = 1
        R0 = 1.2
        Nh = 1e3
        mu = 0.2
        mutationKernel = Normal(0,1)
        mParams = modelParams(r, R0, Nh, mu, mutationKernel)

        translateDistributionBackLeft!(viDist::viralImmuneDistribution, mParams::modelParams)

        @test viDist.nx != nx0
        @test viDist.nx[5*ceil(r)] == nx0[idxPos]
        @test viDist.nx[viDist.nx .== 0] == nx0[nx0 .== 0]
        @test getViralDistribution(viDist.viralPop) == viDist.nx
        @test viDist.hx[1:5*ceil(r)] == hx0[idxPos - 5*ceil(r) + 1:idxPos]
        @test viDist.Reff == begin
            c = conv(viDist.hx, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
            mParams.R0 .* exp.(-c ./ mParams.Nh)
        end

        x = 0:20
        nx0 = zero(x)
        hx0 = zero(x)
        idxVirus = [15, 16, 17]
        nx0[idxVirus] = [1, 3, 2]
        hx0[1:idxVirus[1]] .= 1
        hx0[idxVirus] = [1, 2, 0]
        viDist = viralImmuneDistribution(x, nx0, hx0)

        translateDistributionBackLeft!(viDist::viralImmuneDistribution, mParams::modelParams)

        @test viDist.nx != nx0
        @test viDist.nx[5*ceil(r):5*ceil(r)+2] == nx0[idxVirus]
        @test viDist.nx[viDist.nx .== 0] == nx0[nx0 .== 0]
        @test getViralDistribution(viDist.viralPop) == viDist.nx
        @test viDist.hx[1:5*ceil(r)+2] == hx0[last(idxVirus) - 5*ceil(r) - 2 + 1:last(idxVirus)]
        @test viDist.Reff == begin
            c = conv(viDist.hx, mParams.Hkernel)[mParams.HkernelHalfLength + 1: end - mParams.HkernelHalfLength]
            mParams.R0 .* exp.(-c ./ mParams.Nh)
        end
        
    end

end