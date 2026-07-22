using JLD2

include(expanduser("~/coevolution/code/mutantGrowth/secondMutantStudy.jl"))

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])
s = log(R0)/r

mutationRateVect = let expr = Meta.parse(ARGS[3])
    @assert expr.head == :vect
    Float64.(expr.args)
end
nonLocalMutProb = parse(Float64, ARGS[4])
nonLocalJumpVect = let expr = Meta.parse(ARGS[5])
    @assert expr.head == :vect
    Float64.(expr.args)
end
localKernel = eval(Meta.parse(ARGS[6])) # Dangerous

Nh = parse(Int, ARGS[7])
tmax = parse(Float64, ARGS[8])
totalRuns = parse(Int, ARGS[9])

saveDir = expanduser("~/coevolution/simulations/jumpEffects")
newVMat = Array{Float64}(undef, length(nonLocalJumpVect), length(mutationRateVect))

for i in eachindex(nonLocalJumpVect), j in eachindex(mutationRateVect)

    mutationKernel = piecewiseKernel("piecewise",nonLocalMutProb, nonLocalJumpVect[i], localKernel)
    mutationScale = std(mutationKernel)
    D = mutationRateVect[j] * mutationScale^2 / 2

    saveFile = "jumpEffects_r$(r)R0$(R0)D$(round(D, sigdigits = 2))tmax$(tmax)totalRuns$(totalRuns)Delta$(mutationKernel.nonLocalJump).jld2"
    filePath = joinpath(saveDir, saveFile)
    if isfile(filePath)
        vars = load(filePath)
        newVMat[i, j] = vars["vMod"]
    else
        newVMat[i, j] = NaN
    end
end

plotConfig()
pHeatmap = heatmap(nonLocalJumpVect, mutationRateVect, newVMat', c = cgrad(:magma), xlabel = raw"$\Delta$", ylabel = raw"$\mu$", title = "Average speed of the evolutionary wave", titlefontszie = 20)
pIndividualLines = plot(nonLocalJumpVect, [mutationRateVect[i] .* ones(size(nonLocalJumpVect)) for i in eachindex(mutationRateVect)], [newVMat[:, i] for i in eachindex(mutationRateVect)], cmap = permutedims([palette(:magma, length(mutationRateVect))...]), xlabel = raw"$\Delta$", ylabel = raw"$\mu$", zlabel = raw"$\bar{v}$", framestyle=:axis, grid = true, camera = (10, 40))
p2D = plot(nonLocalJumpVect, [newVMat[:, i] for i in eachindex(mutationRateVect)], cmap = permutedims([palette(:magma, length(mutationRateVect))...]), xlabel = raw"$\Delta$", ylabel = raw"$\bar{v}$")

baseFolder = expanduser("~/coevolution/")
figDir = baseFolder * "figures/macroscopicEffects/"
isdir(figDir) || mkpath(figDir)

savefig(pHeatmap, joinpath(figDir, "speedEffectsHeatmap_r$(r)R0$(R0)tmax$(tmax)totalRuns$(totalRuns).png"))
savefig(pIndividualLines, joinpath(figDir, "speedEffectsIndividualLines_r$(r)R0$(R0)tmax$(tmax)totalRuns$(totalRuns).png"))
savefig(p2D, joinpath(figDir, "speedEffects2D_r$(r)R0$(R0)tmax$(tmax)totalRuns$(totalRuns).png"))