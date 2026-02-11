include("../../code/simulate/coevolution1DSimulationTools.jl")
using SpecialFunctions, LsqFit, LinearAlgebra, NLsolve

function simulateMutantGrowthFull(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x)
    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    # Variable initialisation
    maxIdx = length(x)
    nxBackground = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxBackground[1, :] = nxBackground0
    hx[1, :] = hxBackground0


    # Mutant initialisation
    nxMutant = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxMutant[1, :] = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Instantaneous fields
    hxLoc = hxBackground0
    nxBackgroundLoc = nxBackground0
    nxMutantLoc = nxMutant[1,:]

    for i in 2:length(t)

        # Virus growth
        c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
        R = R0 .* exp.(-c ./ Nh)
        
        nxBackgroundGrowth = rand.(Poisson.(R .* nxBackgroundLoc .* dt))
        nxBackgroundDeath = rand.(Poisson.(nxBackgroundLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        nxBackgroundLoc = max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
        
        nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
        nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
        nxMutantLoc = max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

        nxDeath = nxMutantDeath + nxBackgroundDeath

        # Mutations
        nxBackgroundMutated = sparsevec(rand.(Binomial.(nxBackgroundLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacementsBackground = getDisplacement.(iszero(nxBackgroundMutated) ? [(0, 0)] : tuple.(nxBackgroundMutated.nzind, nxBackgroundMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxBackgroundJump = displacementToJump.(mutationDisplacementsBackground, maxIdx) # 4.643 ms
        nxBackgroundLoc = nxBackgroundLoc - Array(nxBackgroundMutated) + Array(sum(nxBackgroundJump)) # Move mutated viruses
        
        nxMutantMutated = sparsevec(rand.(Binomial.(nxMutantLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacementsMutant = getDisplacement.(iszero(nxMutantMutated) ? [(0, 0)] : tuple.(nxMutantMutated.nzind, nxMutantMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxMutantJump = displacementToJump.(mutationDisplacementsMutant, maxIdx) # 4.643 ms
        nxMutantLoc = nxMutantLoc - Array(nxMutantMutated) + Array(sum(nxMutantJump)) # Move mutated viruses
        
        # Immune evolution
        hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity

        # Sampling
        if i % idxSampling == 1
            nxBackground[Int((i-1)/idxSampling + 1), :] = nxBackgroundLoc
            nxMutant[Int((i-1)/idxSampling + 1), :] = nxMutantLoc
            hx[Int((i-1)/idxSampling + 1), :] = hxLoc
        end
    end 

    println("Simulation end")
    return (nxBackground, nxMutant, hx)
end

function simulateMutantTillFixation(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x; maxItr = 100)
    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    # Variable initialisation
    maxIdx = length(x)
    nxBackground = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxBackground[1, :] = nxBackground0
    hx[1, :] = hxBackground0


    # Mutant initialisation
    nxMutant = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxMutant[1, :] = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Iterations till fixation
    itrTillFixation = 1
    mutantExtinct = false
    backgroundExtinct = false

    for itr in 1:maxItr

        # Instantaneous fields
        hxLoc = hxBackground0
        nxBackgroundLoc = nxBackground0
        nxMutantLoc = nxMutant[1,:]

        println("Iteration $(itr) start")

        for i in 2:length(t)

            # Virus growth
            c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
            R = R0 .* exp.(-c ./ Nh)
            
            nxBackgroundGrowth = rand.(Poisson.(R .* nxBackgroundLoc .* dt))
            nxBackgroundDeath = rand.(Poisson.(nxBackgroundLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
            nxBackgroundLoc = max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
            
            nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
            nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
            nxMutantLoc = max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

            nxDeath = nxMutantDeath + nxBackgroundDeath

            # Mutations
            nxBackgroundMutated = sparsevec(rand.(Binomial.(nxBackgroundLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
            mutationDisplacementsBackground = getDisplacement.(iszero(nxBackgroundMutated) ? [(0, 0)] : tuple.(nxBackgroundMutated.nzind, nxBackgroundMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
            nxBackgroundJump = displacementToJump.(mutationDisplacementsBackground, maxIdx) # 4.643 ms
            nxBackgroundLoc = nxBackgroundLoc - Array(nxBackgroundMutated) + Array(sum(nxBackgroundJump)) # Move mutated viruses
            
            nxMutantMutated = sparsevec(rand.(Binomial.(nxMutantLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
            mutationDisplacementsMutant = getDisplacement.(iszero(nxMutantMutated) ? [(0, 0)] : tuple.(nxMutantMutated.nzind, nxMutantMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
            nxMutantJump = displacementToJump.(mutationDisplacementsMutant, maxIdx) # 4.643 ms
            nxMutantLoc = nxMutantLoc - Array(nxMutantMutated) + Array(sum(nxMutantJump)) # Move mutated viruses

            # Check for mutant extinction
            mutantExtinct = iszero(nxMutantLoc)
            backgroundExtinct = iszero(nxBackgroundLoc)
            if mutantExtinct
                break
            end
            
            # Immune evolution
            hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity

            # Sampling
            if i % idxSampling == 1
                nxBackground[Int((i-1)/idxSampling + 1), :] = nxBackgroundLoc
                nxMutant[Int((i-1)/idxSampling + 1), :] = nxMutantLoc
                hx[Int((i-1)/idxSampling + 1), :] = hxLoc
            end
        end 

        if !mutantExtinct && backgroundExtinct
            itrTillFixation = itr
            println("Mutant fixated at itr $itr")
            break
        else
            println("Mutant extinct")
        end

    end

    println("Simulation end")
    return (nxBackground, nxMutant, hx, mutantExtinct)
end

function simulateMutantTillEstablishment(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x; maxItr = 100, xEstablishment = 0.1)
    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    # Variable initialisation
    maxIdx = length(x)
    nxBackground = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxBackground[1, :] = nxBackground0
    hx[1, :] = hxBackground0


    # Mutant initialisation
    nxMutant = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxMutant[1, :] = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Iterations till fixation
    itrTillEstablishment = 0
    mutantExtinct = false
    backgroundExtinct = false
    mutantEstablished = false

    for itr in 1:maxItr

        # Instantaneous fields
        hxLoc = hxBackground0
        nxBackgroundLoc = nxBackground0
        nxMutantLoc = nxMutant[1,:]

        println("Iteration $(itr) start")

        for i in 2:length(t)

            # Virus growth
            c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
            R = R0 .* exp.(-c ./ Nh)
            
            nxBackgroundGrowth = rand.(Poisson.(R .* nxBackgroundLoc .* dt))
            nxBackgroundDeath = rand.(Poisson.(nxBackgroundLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
            nxBackgroundLoc = max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
            
            nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
            nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
            nxMutantLoc = max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

            nxDeath = nxMutantDeath + nxBackgroundDeath

            # Mutations
            nxBackgroundMutated = sparsevec(rand.(Binomial.(nxBackgroundLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
            mutationDisplacementsBackground = getDisplacement.(iszero(nxBackgroundMutated) ? [(0, 0)] : tuple.(nxBackgroundMutated.nzind, nxBackgroundMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
            nxBackgroundJump = displacementToJump.(mutationDisplacementsBackground, maxIdx) # 4.643 ms
            nxBackgroundLoc = nxBackgroundLoc - Array(nxBackgroundMutated) + Array(sum(nxBackgroundJump)) # Move mutated viruses
            
            nxMutantMutated = sparsevec(rand.(Binomial.(nxMutantLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
            mutationDisplacementsMutant = getDisplacement.(iszero(nxMutantMutated) ? [(0, 0)] : tuple.(nxMutantMutated.nzind, nxMutantMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
            nxMutantJump = displacementToJump.(mutationDisplacementsMutant, maxIdx) # 4.643 ms
            nxMutantLoc = nxMutantLoc - Array(nxMutantMutated) + Array(sum(nxMutantJump)) # Move mutated viruses

            # Check for mutant extinction or establishment
            mutantExtinct = iszero(nxMutantLoc)
            backgroundExtinct = iszero(nxBackgroundLoc)
            if mutantExtinct && !mutantEstablished
                break
            end
            if !mutantEstablished && sum(nxMutantLoc) / sum(nxBackgroundLoc .+ nxMutantLoc) >= xEstablishment
                mutantEstablished = true
            end

            
            # Immune evolution
            hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity

            # Sampling
            if i % idxSampling == 1
                nxBackground[Int((i-1)/idxSampling + 1), :] = nxBackgroundLoc
                nxMutant[Int((i-1)/idxSampling + 1), :] = nxMutantLoc
                hx[Int((i-1)/idxSampling + 1), :] = hxLoc
            end
        end 

        if !mutantExtinct && backgroundExtinct
            itrTillEstablishment = itr
            println("Mutant fixated at itr $itr")
            break
        elseif mutantEstablished && mutantExtinct
            itrTillEstablishment = itr
            println("Mutant established and then extinct")
            break
        elseif mutantEstablished
            itrTillEstablishment = itr
            println("Mutant established")
            break
        else
            println("Mutant extinct")
        end

    end

    println("Simulation end")
    return (nxBackground, nxMutant, hx, mutantExtinct, mutantEstablished, itrTillEstablishment)
end

function countMutantTillNEst(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x, Nfix; maxSearches = 100, maxItrFixation = 100, xEstab = 0.1)
    totalSimulationNumber = 0
    totalEstablishedMutants = 0
    searches = 0
    while (totalEstablishedMutants < Nfix) && (searches < maxSearches)
        println("Dx = $(dxMutant) Search for stablishment #$(searches + 1), found until now $(totalEstablishedMutants) establishment events")
        _ , _ , _ , _, mutantEstablished, simsTillFixation = simulateMutantTillEstablishment(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x, maxItr = maxItrFixation, xEstablishment = xEstab)
        searches += 1
        (simsTillFixation == 0) && (simsTillFixation = maxItrFixation)
        totalSimulationNumber += simsTillFixation
        totalEstablishedMutants += mutantEstablished
    end
    return totalSimulationNumber, totalEstablishedMutants, searches
end

function simulateMutantGrowth(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x)
    # Cross-reactivity Kernel definition
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Instantaneous fields
    hxLoc = hxBackground0
    nxBackgroundLoc = nxBackground0
    nxMutantLoc = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    for i in 2:length(t)

        # Virus growth
        c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
        R = R0 .* exp.(-c ./ Nh)
        
        nxBackgroundGrowth = rand.(Poisson.(R .* nxBackgroundLoc .* dt))
        nxBackgroundDeath = rand.(Poisson.(nxBackgroundLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        nxBackgroundLoc = max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
        
        nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
        nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
        nxMutantLoc = max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

        nxDeath = nxMutantDeath + nxBackgroundDeath

        # Mutations
        nxBackgroundMutated = sparsevec(rand.(Binomial.(nxBackgroundLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacementsBackground = getDisplacement.(iszero(nxBackgroundMutated) ? [(0, 0)] : tuple.(nxBackgroundMutated.nzind, nxBackgroundMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxBackgroundJump = displacementToJump.(mutationDisplacementsBackground, maxIdx) # 4.643 ms
        nxBackgroundLoc = nxBackgroundLoc - Array(nxBackgroundMutated) + Array(sum(nxBackgroundJump)) # Move mutated viruses
        
        nxMutantMutated = sparsevec(rand.(Binomial.(nxMutantLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacementsMutant = getDisplacement.(iszero(nxMutantMutated) ? [(0, 0)] : tuple.(nxMutantMutated.nzind, nxMutantMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        nxMutantJump = displacementToJump.(mutationDisplacementsMutant, maxIdx) # 4.643 ms
        nxMutantLoc = nxMutantLoc - Array(nxMutantMutated) + Array(sum(nxMutantJump)) # Move mutated viruses
        
        # Immune evolution
        hxLoc += nxDeath # Whenever someone recovers it means it has developped immunity
    end 

    println("Simulation end")
    return (nxBackgroundLoc, nxMutantLoc, hxLoc)
end

# =====================================================
#                           Plotting tools
# =====================================================

function animateSimulationMutant(nxBackground, nxMutant, hx, x, Nh)
    
    (absorbedState, idxAbsorbed) = extinctionFlag(nxMutant + nxBackground, x)

    animation = @animate for i in 1:(absorbedState > 0 ? min(idxAbsorbed + 100, size(nxBackground)[1])  : size(nxBackground)[1])
        p = plot(x, nxBackground[i, :], colour=:coral, ylims=[0, max(maximum(nxBackground),maximum(nxMutant))], ylabel=raw"Viral density", xlabel=raw"$x$", label = raw"$n_\mathrm{B}(x,t)$")
        plot!(x, nxMutant[i, :], colour=:indianred, label = raw"$n_\mathrm{M}(x,t)$")
        plot!(twinx(), x, hx[i, :] ./ Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"Immune memories", ylims = [0, 1], label = "")
        plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
    end

    g = gif(animation)
    display(g)

    return g
end

function plotMutantSweep(NtBackground, NtMutant, t)

    NtTotal = NtBackground + NtMutant

    idxMax = findlast(NtTotal .> 0)
    xMutant = NtMutant[1:idxMax] ./ NtTotal[1:idxMax]
    tRange = t[1:idxMax]

    p = plot(tRange, xMutant, colour = :coral, lw = 1.5, ylims = (0,1), fillrange = zero(length(xMutant)), fillalpha = 0.35, yaxis = raw"Mutant fraction" , xaxis = raw"$t$" , label = "")

    return p
end


# =====================================================
#                 Fixation Probability
# =====================================================

function discretisedODE!(f, r, w, dr, vs, Ds)
    f[1] = w[1]
    f[end] = w[end] - r[end]/(1 + r[end])
    f[2:end-1] .= vs/2dr .* (w[3:end] - w[1:end-2]) .- r[2:end-1] .* w[2:end-1] .+ (1 .+ r[2:end-1]) .* w[2:end-1].^2 .- Ds/dr^2 .* (w[3:end] .- 2 .* w[2:end-1] .+ w[1:end-2])
end

alphac(rc, vs, Ds) = vs^2/2Ds - vs/Ds^(1/3)*airyaiprime((vs^2/4Ds - rc)/Ds^(1/3))/airyai((vs^2/4Ds - rc)/Ds^(1/3))
wc(rc, vs, Ds) = (rc-alphac(rc, vs, Ds))/(1+rc)
Ac(rc, vs, Ds) = wc(rc, vs, Ds)/(exp(vs*rc/(2Ds))*airyai((vs^2/4Ds - rc)/Ds^(1/3)))
Cc(rc, vs, Ds) = vs * exp(rc^2/2vs)/wc(rc, vs, Ds) - (vs * exp(rc^2/2vs) + sqrt(pi * vs / 2) * erfi(rc / sqrt(2vs)))
wHighAsymptotic(r, rc, vs, Ds) = vs * exp(r^2 / 2vs) / (vs * exp(r^2 /2vs) + Cc(rc, vs, Ds) + sqrt(pi * vs / 2) * erfi(r / sqrt(2vs)))
wLowAsymptotic(r, rc, vs, Ds) = Ac(rc, vs, Ds) * airyai((vs^2/4Ds-r)/Ds^(1/3)) * exp(vs*r/2Ds)
wAsymptotic(r, rc, vs, Ds) = wHighAsymptotic.(r, rc, vs, Ds) .* (r .> rc) .+ wLowAsymptotic.(r, rc, vs, Ds) .* (r .< rc)

function getFixedPoint(vs, Ds, zeta, d)
    rc0 = zeta + 1.6d
    funToZero(r) = alphac.(r, vs, Ds) .- r
    sol = nlsolve(funToZero, [rc0])
    return (sol.zero[1], (sol.zero[1] - zeta) / d)
end

function getLowFitnessMaximum(vs, Ds, zeta, d)
    rc0 = zeta + 1.6d
    funToZero(r) = alphac.(r, vs, Ds)
    sol = nlsolve(funToZero, [rc0])
    return (sol.zero[1], (sol.zero[1] - zeta) / d)
end

function getMatchingLimits(vs, Ds)
    zeta = vs^2/4Ds
    d = Ds^(1/3)

    (rFixed, xiFixed) =  getFixedPoint(vs, Ds, zeta, d)
    (rM, xiM) = getLowFitnessMaximum(vs, Ds, zeta, d)

    return (rFixed, xiFixed, rM, xiM)
end

function getNumericalFixationProbability(rVect, dr, vs, Ds, rc0)    
    zeta = vs^2/4Ds
    d = Ds^(1/3)

    w0 = wAsymptotic(rVect, rc0, vs, Ds)
    w0[w0 .< 0] = 0
    w0[isnan.(w0)] = rVect./(1 .+ rVect)[isnan.(w0)]
    w0[1] = 0

    discretisedODEToSolve! = (f,w) -> discretisedODE!(f, rVect, w, dr, vs, Ds)
    sol = nlsolve(discretisedODEToSolve!, w0)
    return sol.zero
end

function getBestMatchingPoint(wNumeric, vs, Ds, rVect, rc0)

    zeta = vs^2/4Ds
    d = Ds^(1/3)

    fittingFunc(r, p) = wAsymptotic(r, p[1], vs, Ds)
    fit = curve_fit(fittingFunc, rVect, wNumeric, [rc0])
    rcFit = fit.param[1]
    xiFit = (rcFit - zeta)/d
    
    return (rcFit, xiFit)
end