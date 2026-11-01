include("../../src/viralImmune2D.jl")

using LinearAlgebra

function getRandomImmuneLandscape(originalR0, targetR0, landSize, Hkernel, Nh; dH=Nh÷5)
    Reff = originalR0 * ones(landSize)
    HField = zeros(Int, landSize)
    cartesianIdxs = CartesianIndices(HField)
    lowerIdx, higherIdx = minimum(cartesianIdxs), maximum(cartesianIdxs)
    sizeHkernel = size(Hkernel)

    while mean(Reff) > targetR0
        normalizedCumProb = cumsum(normalize(Reff ./ (1 .+ Reff), 1)[:])
        zeta = rand()
        idx = cartesianIdxs[findfirst(normalizedCumProb .>= zeta)]
        global HField[idx] = HField[idx] + dH
        minIdx = clampIdx(CartesianIndex(idx[1]-(sizeHkernel[1]-1)÷2, idx[2]-(sizeHkernel[2]-1)÷2), lowerIdx, higherIdx)
        maxIdx = clampIdx(CartesianIndex(idx[1]+(sizeHkernel[1]-1)÷2, idx[2]+(sizeHkernel[2]-1)÷2), lowerIdx, higherIdx)
        fieldIdx = minIdx:maxIdx
        @views Reff[fieldIdx] .*= exp.(- dH .* Hkernel[fieldIdx .- idx .+ CartesianIndex((sizeHkernel[1]-1)÷2+1, (sizeHkernel[2]-1)÷2+1)] ./ Nh)
    end

    return Reff, HField
end


function plotAnimationReff(sampDist::sampledResults, x::Union{UnitRange,Nothing}=nothing, y::Union{UnitRange,Nothing}=nothing)

    isempty(sampDist.nxt) && throw(ArgumentError("The given sampled distributions are empty"))

    isnothing(x) && (x = 1:size(first(sampDist.nxt))[1])
    isnothing(y) && (y = 1:size(first(sampDist.nxt))[2])

    idxs = CartesianIndices(first(sampDist.nxt))

    maxN = maximum(hcat(sampDist.nxt...))
    minN = 0
    maxReff = maximum(hcat(sampDist.Refft...))
    minReff = minimum(hcat(sampDist.Refft...))
    NscalingFactor = (maxReff-minReff)/maxN

    g = @gif for i in 1:length(sampDist.nxt)
        l = @layout [a{0.8w} b{0.1w} c{0.1w}]
        hp = heatmap(x, y, sampDist.Refft[i]', zlabel=raw"$h(x,y)$", xlabel=raw"$x$", ylabel=raw"$y$", clims=(minReff, maxReff), c=cgrad(:inferno, scale=:linear), cbar=:none)
        heatmap!(hp, x, y, replace(sampDist.nxt[i], 0 => NaN)' .* NscalingFactor .+ minReff, zlabel=raw"$n(x,y)$", c=cgrad(:winter, scale=:exp), cbar=:none)
        cbar1 = heatmap(rand(2, 2), clims=(minReff, maxReff), framestyle=:none, c=cgrad(:inferno, scale=:linear), cbar=true, lims=(-1, 0))
        cbar2 = heatmap(rand(2, 2), clims=(minN, maxN), framestyle=:none, c=cgrad(:winter, scale=:exp), cbar=true, lims=(-1, 0))
        plot(hp, cbar1, cbar2, layout=l, rigth_margin=20Plots.px)
    end

    display(g)
    return g
end