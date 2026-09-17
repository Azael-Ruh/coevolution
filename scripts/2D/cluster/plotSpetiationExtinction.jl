include("../src/viralImmune2D.jl")
using JLD2

r = parse(Int, ARGS[1])
R0 = parse(Float64, ARGS[2])

mutationRateVect = let expr = Meta.parse(ARGS[3])
    @assert expr.head == :vect
    Float64.(expr.args)
end
nonLocalMutProb = Meta.parse(ARGS[4])
nonLocalJumpVect = let expr = Meta.parse(ARGS[5])
    @assert expr.head == :vect
    Int.(expr.args)
end
localMutationKernel = NextNeighbour()

Nh = parse(Int, ARGS[6])
tmax = parse(Float64, ARGS[7])
nCycles = parse(Int, ARGS[8])
runs = parse(Int, ARGS[9])

saveDir = expanduser("~/coevolution/simulations/2D/speciationNExtinction")

speciationTimes = zeros(length(mutationRateVect), length(nonLocalJumpVect))
nSpeciationEvents = Int.(speciationTimes)
extinctionTimes = zeros(length(mutationRateVect), length(nonLocalJumpVect))
nExtinctionEvents = Int.(speciationTimes)
nonLocalTimes = zeros(length(mutationRateVect), length(nonLocalJumpVect))
nNonLocalEvents = Int.(speciationTimes)

for i in eachindex(mutationRateVect), j in eachindex(nonLocalJumpVect)  
    for run in 1:runs
        saveFile = "speciationExtinction_r$(r)R0$(R0)mu$(mutationRateVect[i])tmax$(tmax)nCycles$(nCycles)nonLocalProb$(nonLocalMutProb)Delta$(nonLocalJumpVect[j])_run$(run).jld2"
        filePath = joinpath(saveDir, saveFile)
        
            println("Loofing for file $(filePath)!")
        if isfile(filePath)
            println("Found file!")
            vars = load(filePath)
            nSpeciationEvents[i,j] += vars["nSpeciationEvents"]
            nExtinctionEvents[i,j] += vars["nExtinctionEvents"]
            nNonLocalEvents[i,j] += vars["nNonLocalEvents"]
        end 
    end

    nNonLocalEvents[i,j] == 0 && (nSpeciationEvents[i,j] = -1; nExtinctionEvents[i,j] = -1; nNonLocalEvents[i,j] = -1)
end

saveFile = "speciationExtinctionMatrixes_r$(r)R0$(R0)nonLocalProb$(nonLocalMutProb)tmax$(tmax)nCycles$(nCycles).jld2"
filePath = joinpath(saveDir, saveFile)
jldsave(filePath; nSpeciationEvents, nExtinctionEvents, nNonLocalEvents)