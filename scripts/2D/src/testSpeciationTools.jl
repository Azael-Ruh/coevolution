using Test, DSP

include("./speciationTools.jl")

@testset  "speciationTools" begin

    @testset "constructors" begin

        N = 10
        Rmean = 3
        logSurv = 3* 0.5
        rmean = [0, 0]
        species = viralSpecies(N, Rmean, rmean, logSurv)

        @test typeof(species) == viralSpecies
        @test species.N == N
        @test species.Rmean == Rmean
        @test species.logSurvivability == logSurv 
        @test species.rmean == rmean 
        @test isnothing(species.nxMasked) 

        N = 0

        @test_throws ArgumentError viralSpecies(N, Rmean, rmean, logSurv)

        Rmean = 0
        N = 1
        
        @test_throws ArgumentError viralSpecies(N, Rmean, rmean, logSurv)

        Rmean = 1
        nxMask = ones(10, 10)
        species = viralSpecies(N, Rmean, logSurv, rmean, nxMask)

        @test species.nxMasked == nxMask
    end

    @testset "labelling" begin
        
        R0 = 1.1
        r = 1
        Nh = 1000

        x = -10:10
        y = -10:10
        nx::Matrix{Int} = zeros(length(x), length(y))

        gaussianCond2D(x, y, x0, y0, sig) = exp(-((x-x0)^2 + (y-y0)^2)/(2 * sig^2)) / sqrt(2*pi*sig^2)^2

        blob1 = round.(Int, 150 .* [gaussianCond2D(xel, yel, -5, -5, 2) for xel in x, yel in y])
        blob2 = round.(Int, 100 .* [gaussianCond2D(xel, yel, -3, 6, 2) for xel in x, yel in y])
        blob3 = round.(Int, 100 .* [gaussianCond2D(xel, yel, 6, 3, 2) for xel in x, yel in y])

        nx += blob1 + blob2 + blob3

        viralLabels = getViralLabels(nx, r)

        @test maximum(viralLabels) == 3
        @test sum(nx[viralLabels .== 1]) > 100
        @test sum(nx[viralLabels .== 2]) > 50 && sum(nx[viralLabels .== 3]) > 50

        Reff = ones(size(nx))
        speciesValues = getSpeciesValues(viralLabels, nx, x, y, Reff)

        @test speciesValues[1].Rmean == speciesValues[2].Rmean == speciesValues[3].Rmean == 1
        @test speciesValues[1].N == sum(blob1)
        @test speciesValues[2].N == sum(blob2)
        @test speciesValues[3].N == sum(blob3)
        @test speciesValues[1].rmean == [-5, -5]
        @test speciesValues[2].rmean == [6, 3]
        @test speciesValues[3].rmean == [-3, 6]
    end

    @testset "selection" begin

        R0 = 1.1
        r = 1
        Nh = 1000

        x = -100:100
        y = -100:100
        nx::Matrix{Int} = zeros(length(x), length(y))

        gaussianCond2D(x, y, x0, y0, sig) = exp(-((x-x0)^2 + (y-y0)^2)/(2 * sig^2)) / sqrt(2*pi*sig^2)^2

        blob1 = round.(Int, 150 .* [gaussianCond2D(xel, yel, -5, -5, 2) for xel in x, yel in y])
        blob2 = round.(Int, 100 .* [gaussianCond2D(xel, yel, -3, 6, 2) for xel in x, yel in y])
        blob3 = round.(Int, 100 .* [gaussianCond2D(xel, yel, 6, 3, 2) for xel in x, yel in y])

        nx += blob1 + blob2 + blob3

        viralLabels = getViralLabels(nx, r)

        idxZero = 101

        immunePath01(t) = idxZero .+ [-10 + 5t, 0]
        immunePath13_1(t) =  idxZero .+ [-5, - 5(t - 1) / 2]
        immunePath12_2(t) = idxZero .+ [-5 + 2(t - 1), 3(t - 1)]
        immunePath23_21(t) = idxZero .+ [-3, 3(t - 1)]
        immunePath23_22(t) = idxZero .+ [-3 + 9(t - 2), 3]

        hx = zero(nx)
        [(hx[round.(Int, immunePath01(t))...] += 1) for t in 0:0.005:1]
        [(hx[round.(Int, immunePath13_1(t))...] += 1) for t in 1:0.004:3]
        [(hx[round.(Int, immunePath12_2(t))...] += 1) for t in 1:0.005:2]
        [(hx[round.(Int, immunePath23_21(t))...] += 1) for t in 2:0.005:3]
        [(hx[round.(Int, immunePath23_22(t))...] += 1) for t in 2:0.005:3]

        H(x) = exp.(-abs.(x)/r)
        r == 0 ? (Hkernel = ones(1,1)) : begin
            HkernelSpace = -5*ceil(r):5*ceil(r)
            Hkernel = H(sqrt.(HkernelSpace.^2 .+ HkernelSpace'.^2))
        end
        HkernelHalfLength::Int = floor(size(Hkernel)[1]/2)

        c = conv(hx, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength, HkernelHalfLength + 1: end - HkernelHalfLength]
        Reff = R0 .* exp.(-c ./ Nh)

        speciesValues = getSpeciesValues(viralLabels, nx, x, y, Reff, withMask = true)
        bestSpecies = getBestSpecies(speciesValues)

        @test bestSpecies == speciesValues[2]

        maskedNx = getMaskedDistribution(bestSpecies)

        @test maskedNx == blob3
    end

    

end