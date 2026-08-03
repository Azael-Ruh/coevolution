include("../../../code/simulate/coevolution1DSimulationTools.jl")

using DataStructures

function getDisplacementVariant(nMutated::Tuple{Int64, Int64}, mutKernel::piecewiseKernel)
    getDisplacementVariant(nMutated, kernType(mutKernel), mutKernel.nonLocalMutProb, mutKernel.nonLocalJump, mutKernel.localKernel)
end

function getDisplacementVariant(nMutated::Tuple{Int64, Int64}, mutKernel::String, longJumpProb = 0., longJumpLength = 0, localKernel = Normal(0, 1))
    nMutated[1] == 0 && return (0,[0])
    if mutKernel == "piecewise" 
        nNonLocal = rand(Binomial(nMutated[2], longJumpProb))
        return (nMutated[1], Distributions.rand!(localKernel, zeros(nMutated[2] - nNonLocal)), longJumpLength .* ones(nNonLocal))
    end
    error("Distribution not yet implemented or mispelled")
end

function simulateVariants()

    println("Simulation end")
    return (nxBackground, nxMutant, hx)


end

function animateVariantSimulationLog(nxVect, hx, x, Nh)

    nLogMax = maximum([max([maximum(log.(nxEl)) for nxEl in nxVect[i]]...) for i in eachindex(nxVect)])

    animation = @animate for i in 1:length(nxVect)
        p = plot(x, reverse([replace(log.(nxEl), -Inf => -1) for nxEl in nxVect[i]]), ylims=[-1, nLogMax], ylabel=raw"log(Viral density)", xlabel=raw"$x$", label = "")
        plot!(twinx(), x, hx[i, :] ./ Nh, colour = :steelblue, background_color_legend = :white, yaxis = raw"log(Immune memories)", ylims = [0, 0.5], label = "")
        plot!([], [], color = :coral, label = raw"$\log\,n(x,t)$")
        plot!([], [], color = :steelblue, label = raw"$h(x,t)/N_h$", legend_pos = :topright)
    end

    g = gif(animation)
    display(g)

    return g
end



r = 50
R0 = 1.3
s = log(R0)/r

mutationRate = 0.30
nonLocalMutProb = 1e-6
nonLocalJump = 100
localKernel = Normal(0,1)
mutationKernel = piecewiseKernel("piecewise", nonLocalMutProb, nonLocalJump, localKernel)

mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2

Nh::Int64 = 1e8

tmax = 900
vFKKP = 2 .* sqrt.((R0 .- 1) .* D)
xmax = 2*max(500, round(2 .* vFKKP .* tmax, digits = -2))

(nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

dt = 0.1
dtSampling = 3

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
variantList = Array{Cons{Tuple{Int64, Vector{Float64}}}, 1}(undef, Int(round(tmax/dtSampling+1)))
hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
hx[1, :] = hx0

println("=============START OF THE SIMULATION==============")
t = 0:dt:tmax
idxSampling::Int = round(dtSampling/dt)

# Instantaneous fields
hxLoc = copy(hx0)
nxBackground = copy(nx0)
VlistLoc = list((1, nxBackground))

variantList[1] = copy(VlistLoc)
variantIdx = 1

for i in 2:length(t)

    # Virus growth
    c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
    R = R0 .* exp.(-c ./ Nh)

    currentNode = VlistLoc
    nTotalDeath = zero(hxLoc)
    while !isempty(currentNode)

        currentIdx = first(head(currentNode))
        nxLoc = last(head(currentNode))

        nxGrowth = rand.(Poisson.(R .* nxLoc .* dt))
        nxDeath = rand.(Poisson.(nxLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        nxLoc = max.(nxLoc .+ nxGrowth .- nxDeath, 0)

        nTotalDeath += nxDeath

        # Mutations
        nxMutated = sparsevec(rand.(Binomial.(nxLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacementsVariant = getDisplacementVariant.(iszero(nxMutated) ? [(0, 0)] : tuple.(nxMutated.nzind, nxMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        
        # Local Mutations
        mutationDisplacementsLocal = [mutDisp == (0, [0]) ? (0, [0]) : (first(mutDisp), mutDisp[2]) for mutDisp in mutationDisplacementsVariant]
        nxJumpLocal = displacementToJump.(mutationDisplacementsLocal, maxIdx) # 4.643 ms
        nxMutationMovedLocal = Array(sum(nxJumpLocal))
        nxLoc = nxLoc - Array(nxMutated) + Array(sum(nxJumpLocal)) # Move mutated viruses            
           
        # Node update
        currentNode.head = (currentIdx, nxLoc)

        # nonLocal Mutations
        mutationDisplacementsNonLocal = [mutDisp == (0, [0]) ? (0, [0]) : (first(mutDisp), mutDisp[3]) for mutDisp in mutationDisplacementsVariant]
        nxJumpNonLocal = displacementToJump.(mutationDisplacementsNonLocal, maxIdx) # 4.643 ms
        newVariants = Array(sum(nxJumpNonLocal))

        idxVariants = findall(newVariants .> 0)
        for i in eachindex(idxVariants)
            global variantIdx += 1
            VlistLoc.tail = cons((variantIdx, newVariants[idxVariants[i]]*(x .== x[idxVariants[i]])), VlistLoc.tail)
            println("Variant added")
        end

        currentNode = currentNode.tail
    end

    # Eliminate dead variants
    variantsBefore = length(VlistLoc)
    global VlistLoc = filter(x -> !iszero(last(x)), VlistLoc)
    variantsNow = length(VlistLoc)
    variantsNow < variantsBefore && println("Eliminated $(variantsBefore - variantsNow) variants.")
    
    # Immune evolution
    global hxLoc += nTotalDeath # Whenever someone recovers it means it has developped immunity

    if isempty(VlistLoc)
        println("WARNING: total extinction")
        [variantList[k] = variantList[Int(ceil((i-1)/idxSampling + 1)) - 1] for k in Int(ceil((i-1)/idxSampling + 1)):length(variantList)]
        [hx[k,:] = copy(hxLoc) for k in Int(ceil((i-1)/idxSampling + 1)):length(variantList)]
        break
    end

    if i % idxSampling == 1
        variantList[Int((i-1)/idxSampling + 1)] = copy(VlistLoc)
        hx[Int((i-1)/idxSampling + 1), :] = copy(hxLoc)
    end

end 

nxVect = [last.(el) for el in collect.(variantList)]
idxVect = [first.(el) for el in collect.(variantList)]

g = animateVariantSimulationLog(nxVect, hx, x, Nh)