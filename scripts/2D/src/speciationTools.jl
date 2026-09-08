using ImageMorphology

# =======================================================
#               Structs and constructors
# =======================================================

struct viralSpecies
    label::Int
    N::Int
    Rmean::Real
    logSurvivability::Real
    rmean::Vector{<:Real}
    nxMasked::Union{Matrix{<:Integer}, Nothing}
    #TODO: delete below if not used in the end
    # nxMask::Union{Matrix{Bool}, Nothing}
    viralSpecies(label::Int, N::Int, Rmean::Real, surv::Real, rmean::Vector{<:Real}, nxMask::Union{Matrix{<:Integer}, Nothing}) = begin
        label >= 0 || throw(ArgumentError("The label should be non-negative"))
        N > 0 || throw(ArgumentError("N should be positive."))
        Rmean > 0 || throw(ArgumentError("The average growth rate should be positive"))
        # isnothing(nxMask) ?
        #     new(N, Rmean, surv, nxMask, nothing) :
        #     new(N, Rmean, surv, nxMask, nxMask .> 0)
        new(label, N, Rmean, surv, rmean, nxMask)
    end
end

viralSpecies(label::Int, N::Int, Rmean::Real, surv::Real, rmean::Vector{<:Real}, nxMask::Matrix{Float64}) = viralSpecies(label, N, Rmean, surv, rmean, convert(Matrix{Int}, nxMask))

viralSpecies(N::Int, Rmean::Real, surv::Real, rmean::Vector{<:Real}, nxMask::Union{Matrix{<:Integer}, Nothing}) = viralSpecies(1, N, Rmean, surv, rmean, nxMask)

viralSpecies(N::Int, Rmean::Real, surv::Real, rmean::Vector{<:Real}, nxMask::Matrix{Float64}) = viralSpecies(1, N, Rmean, surv, rmean, convert(Matrix{Int}, nxMask))

viralSpecies(label::Int, N::Int, Rmean::Real, rmean::Vector{<:Real}, surv::Real) = viralSpecies(N, Rmean, surv, rmean, nothing)

viralSpecies(N::Int, Rmean::Real, rmean::Vector{<:Real}, surv::Real) = viralSpecies(1, N, Rmean, surv, rmean, nothing)

logSurvivability(species::viralSpecies) = species.logSurvivability

# =======================================================
#               Private functions
# =======================================================

function getViralLabels(nx::Matrix{<:Integer}, r::Real)
    nxMask = nx .> 0
    interactionRange::Int = ceil(Int, max(r, 1))
    neighbourmask = [norm([i,j]) <= interactionRange for i in -interactionRange:interactionRange, j in -interactionRange:interactionRange]
    return label_components(nxMask, neighbourmask)
end

function getSpeciesValues(nxComponents::Matrix{Int}, nx::Matrix{<:Integer}, x::UnitRange, y::UnitRange, Reff::Matrix{<:Real}; withMask::Bool = false)
    nSpecies = maximum(nxComponents)
    speciesVect = viralSpecies[]
    for species = 1:nSpecies
        speciesMask = nxComponents .== species
        nxMasked = nx .* speciesMask
        
        N = sum(nxMasked)
        Rmean = sum(nxMasked .* Reff) / N
        logSurvivability  = - sum(nxMasked .* log.(1 ./ Reff))

        rmean = [sum(x' * nxMasked), sum(nxMasked * y)] ./ N

        withMask ? 
            push!(speciesVect, viralSpecies(species, N, Rmean, logSurvivability, rmean, nx .* speciesMask)) : 
            push!(speciesVect, viralSpecies(species, N, Rmean, rmean, logSurvivability))
    end
    return speciesVect
end

getBestSpecies(speciesVect::Vector{viralSpecies}) = speciesVect[argmax(logSurvivability.(speciesVect))]

function getMaskedDistribution(species::viralSpecies)
    return species.nxMasked
end

function getMaskedDistribution(species::viralSpecies, viralLabels::Matrix{Int}, nx::Matrix{<:Integer})
    return nx .* (viralLabels .== species.label)
end