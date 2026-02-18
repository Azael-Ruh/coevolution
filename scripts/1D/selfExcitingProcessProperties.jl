include("../../code/simulate/coevolution1DSimulationTools.jl")

using Peaks

function getPeaks(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x, NAv, Nstd)
    Nt, xt = simulateWaveMacro(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x)

    peaks = findmaxima(Nt, 10)
    filterpeaks!(peaks, :heights; min = NAv + 2.5Nstd)

    # peaks = []
    # lastIdx = 1
    # while lastIdx < length(NtPeak) && !isnothing(findfirst(NtPeak[lastIdx:end] .> 0))
    #     peakIdx = findfirst(NtPeak[lastIdx:end] .> 0) + lastIdx - 1
    #     lastIdx = (isnothing(findfirst(NtPeak[peakIdx:end] .== false)) ? length(NtPeak) : findfirst(NtPeak[peakIdx:end] .== false) + peakIdx - 1)
    #     println("($peakIdx, $lastIdx)")
    #     append!(peaks, (peakIdx + lastIdx - 1)/2)
    # end
    
    return peaks, Nt
end


Nh::Int64 = 10000000

r = 2
R0 = 1.2

mutationRate = 0.5
nonLocalMutationProb = 1e-7
localKernel = Normal(0,1)

s = log(R0)/r
D = mutationRate*std(localKernel)^2*(1-nonLocalMutationProb)/2

vFKPP = 2 * sqrt((R0 - 1) * D)
tmax = 3000
xmax = 2*max(500, round(Int, vFKPP*tmax + vFKPP^2/D))

dt = 0.1
dtSampling = 1
tTransient = 100

NAv, Nstd, vAv, sigmaAv, uTAv = simulateWaveStatisticsFull(R0, r, Nh, mutationRate, localKernel, dt, tmax, dtSampling, tTransient)

nonLocalJumpGrid = [10:5:20; 22:2:40; 45:5:60]
nPeaksVect = []
deltaTVect = []

for nonLocalJump in nonLocalJumpGrid

    global mutationKernel = picewiseKernel("piecewise", nonLocalMutationProb, nonLocalJump, localKernel)

    (nx0, hx0, x) = getInitialCondition("steadyState", R0, r, mutationRate, mutationKernel, Nh, xmax)

    Ntimes = 25
    deltaT = []
    nPeaks = []
    Nt = []
    peaks = []
    println("Simulating for Delta = $(nonLocalJump)")
    for i = 1:Ntimes
        peaks, Nt = getPeaks(nx0, hx0, R0, r, Nh, mutationRate, mutationKernel, dt, tmax, dtSampling, x, NAv, Nstd)
        peakPositions = peaks.indices
        append!(deltaT, peakPositions[2:end] - peakPositions[1:end-1])
        append!(nPeaks, length(peakPositions))
    end

    push!(nPeaksVect, nPeaks)
    push!(deltaTVect, deltaT)

    plotConfig()
    t = 0:dtSampling:tmax
    p = plot(t, Nt, colour = :coral, xlabel = raw"$t$", ylabel = raw"N(t)")
    scatter!(peaks.indices, peaks.heights)
    display(p)
end

meanPeaks = mean.(nPeaksVect)
stdPeaks = std.(nPeaksVect)
clusterRatio = stdPeaks.^2 ./meanPeaks
plot(nonLocalJumpGrid, clusterRatio)