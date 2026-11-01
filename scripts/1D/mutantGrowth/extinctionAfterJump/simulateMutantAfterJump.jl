include("./src/viralImmuneDistribution.jl")

using LinearAlgebra

R0::Float64 = 1.2
r::Int = 40
mu::Float64 = 0.03
x0::Int = 42 #40
mutationKernel = Normal(0,1)
Nh::Int = 1e8
mParams = modelParams(r, R0, Nh, mu, mutationKernel)

x = 0:400
tmax = 500
dt = 0.1
dtSampling = 0.2
simSet = simulationConfig(tmax, dt, dtSampling)
extinctionTimes = Float64[]

viDist = newlySeededDistribution(x, x0)
getGrowthRateLinear!(viDist, mParams)
sampDists = sampledResults(viDist)

NtVect = Vector{Int}[]
uTVect = Vector{Int}[]
sigmaVect = Vector{Float64}[]
xtVect = Vector{Float64}[]
sTVect = Vector{Float64}[]

extinctionFlag = true
times = 0:dt:tmax

runs = 100
while extinctionFlag || (runs > 0)
    global sampDists = sampledResults(viDist)
    global viDist = newlySeededDistribution(x, x0)
    getGrowthRateLinear!(viDist, mParams)
    sampleDistributions!(viDist, sampDists)

    for t in times[2:end]
        global extinctionFlag = simulationStep!(viDist, mParams, simSet) 
        
        extinctionFlag && (println("WARNING: virus extinct"); push!(extinctionTimes, t); break)
        
        if isTime2Sample(t, simSet)
            println("t = $t")
            sampleDistributions!(viDist, sampDists)
        end
    end

    push!(NtVect, sum.(sampDists.nxt))
    lastIdxt = findlast.(x -> x > 0, sampDists.nxt)
    push!(uTVect, x[lastIdxt])
    push!(sTVect, [sampDists.Refft[i][lastIdxt[i]] for i in eachindex(lastIdxt)])
    push!(xtVect, dot.(sampDists.nxt, [viDist.space]) ./ NtVect[end])
    push!(sigmaVect, dot.(sampDists.nxt, [viDist.space.^2]) ./ NtVect[end] .- xtVect[end].^2)

    extinctionFlag || (global runs -= 1; println("========= $runs runs left! ========="))
end

simTimes = [(0:length(uTt).-1).*dtSampling for uTt in uTVect]
maxlength = maximum(length.(uTVect))

pxT = plot(ylabel = raw"$x_T$");
plot!.([pxT], filter(x -> maxlength == length(x), simTimes), filter(x -> maxlength == length(x), uTVect), c = :gray, alpha = 0.2, label = "")
plot!.([pxT], filter(x -> maxlength > length(x) > 200, simTimes), filter(x -> maxlength > length(x) > 200, uTVect), c = :red, alpha = 0.2, label = "")
display(pxT)

puT = plot(ylabel = raw"$u_T$");
plot!.([puT], filter(x -> maxlength == length(x), simTimes), filter(x -> maxlength == length(x), uTVect .- xtVect), c = :gray, alpha = 0.2, label = "")
plot!.([puT], filter(x -> maxlength > length(x) > 200, simTimes), filter(x -> maxlength > length(x) > 200, uTVect .- xtVect), c = :red, alpha = 0.2, label = "")
display(puT)

idxMax = argmax.([Nt[1:min(600,end)] for Nt in NtVect])
xMax = [ones(length(xtVect[i])).*xtVect[i][idxMax[i]] for i in eachindex(idxMax)]

puTxMax = plot(ylabel = raw"$x_T - x^*$");
plot!.([puTxMax], filter(x -> maxlength == length(x), simTimes), filter(x -> maxlength == length(x), uTVect .- xMax), c = :gray, alpha = 0.2, label = "")
plot!.([puTxMax], filter(x -> maxlength > length(x) > 200, simTimes), filter(x -> maxlength > length(x) > 200, uTVect .- xMax), c = :red, alpha = 0.2, label = "")
display(puTxMax)

psT = plot(ylabel = raw"$s_T$");
plot!.([psT], filter(x -> maxlength == length(x), simTimes), filter(x -> maxlength == length(x), sTVect), c = :gray, alpha = 0.2, label = "")
plot!.([psT], filter(x -> maxlength > length(x) > 200, simTimes), filter(x -> maxlength > length(x) > 200, sTVect), c = :red, alpha = 0.2, label = "")
display(psT)

elemLog(x) = log.(x)

pNt = plot(ylabel = raw"$\log{N}$");
plot!.([pNt], filter(x -> maxlength == length(x), simTimes), elemLog.(filter(x -> maxlength == length(x), NtVect)), c = :gray, alpha = 0.2, label = "")
plot!.([pNt], filter(x -> maxlength > length(x) > 200, simTimes), elemLog.(filter(x -> maxlength > length(x) > 200, NtVect)), c = :red, alpha = 0.2, label = "")
display(pNt)

pxT = plot(ylabel = raw"$\bar{x}$");
plot!.([pxT], filter(x -> maxlength == length(x), simTimes), filter(x -> maxlength == length(x), xtVect), c = :gray, alpha = 0.2, label = "")
plot!.([pxT], filter(x -> maxlength > length(x) > 200, simTimes), filter(x -> maxlength > length(x) > 200, xtVect), c = :red, alpha = 0.2, label = "")
display(pxT)