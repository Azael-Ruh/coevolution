include("../../code/simulate/coevolution1DSimulationTools.jl")

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
    hx[1, :] = hx0


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
        nxBackgroundLoc .= max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
        
        nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
        nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
        nxMutantLoc .= max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

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
    hx[1, :] = hx0


    # Mutant initialisation
    nxMutant = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxMutant[1, :] = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Iterations till fixation
    itrTillFixation = 1
    mutantExtinct = false

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
            nxBackgroundLoc .= max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
            
            nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
            nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
            nxMutantLoc .= max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

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

        if !mutantExtinct
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

function simulateMutantTillEstablishment(nxBackground0, hxBackground0, dxMutant, r, R0, mutationRate, mutationKernel, Nh, tmax, dt, dtSampling, x; maxItr = 100, Nestablishment = 100)
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
    hx[1, :] = hx0


    # Mutant initialisation
    nxMutant = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
    nxMutant[1, :] = (x .== round(Int, sum(x .* nxBackground0) / sum(nxBackground0) + dxMutant))

    println("=============START OF THE SIMULATION==============")
    t = 0:dt:tmax
    idxSampling::Int = round(dtSampling/dt)

    # Iterations till fixation
    itrTillFixation = 1
    mutantExtinct = false
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
            nxBackgroundLoc .= max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
            
            nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
            nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
            nxMutantLoc .= max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

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
            if mutantExtinct && !mutantEstablished
                break
            end
            if !mutantEstablished && sum(nxMutantLoc) >= Nestablishment
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

        if !mutantExtinct
            itrTillFixation = itr
            println("Mutant fixated at itr $itr")
            break
        elseif mutantEstablished
            println("Mutant established and then extinct")
            break
        else
            println("Mutant extinct")
        end

    end

    println("Simulation end")
    return (nxBackground, nxMutant, hx, mutantExtinct, mutantEstablished)
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
        nxBackgroundLoc .= max.(nxBackgroundLoc .+ nxBackgroundGrowth .- nxBackgroundDeath, 0)
        
        nxMutantGrowth = rand.(Poisson.(R .* nxMutantLoc .* dt))
        nxMutantDeath = rand.(Poisson.(nxMutantLoc .* dt))
        nxMutantLoc .= max.(nxMutantLoc .+ nxMutantGrowth .- nxMutantDeath, 0)

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