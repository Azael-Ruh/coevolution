# ==============================================================
#   Simulate1DWave.jl
# ==============================================================
#
# Project: punctuatedCoevolution (M2 Internship - PhD)
# Author: Max Zayas Orihuela (maxzyas@gmail.com)
# Supervisors: Aleksandra Walczak, Thierry Mora
# Last update: 19/05/2025
# Purpose: produce a 1D simulation of the evolution of an antigenic wave for a rapidly mutating virus under the pressure of the immune system within a population of hosts. The objective of the simulation is to study the effect of non-local mutations.


# ===================================================
#   Imports
# ===================================================
using Distributions, Plots, StatsBase, Interpolations, Tables, CSV, SparseArrays, DSP, SpecialFunctions, NLsolve

# ====================================================================
#                   Useful functions
# ====================================================================

# General case
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Distribution{Univariate, Continuous})
    nMutated[1] == 0 && return (0,[0])
    (nMutated[1], Distributions.rand!(mutKernel, zeros(nMutated[2])))
end

# Exponential case
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Exponential{Float64})
    nMutated[1] == 0 && return (0,[0])
    mutSign = (Distributions.rand!(Binomial(1, 0.5), zeros(nMutated[2])) .- 0.5) .* 2
    (nMutated[1], Distributions.rand!(mutKernel, zeros(nMutated[2])) .* mutSign)
end

function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::Distribution{Univariate, Continuous}, longJumpProb, longJumpLength, localKernel)
    getDisplacement(nMutated, mutKernel)
end

# Special distributions
function getDisplacement(nMutated::Tuple{Int64, Int64}, mutKernel::String, longJumpProb = 0., longJumpLength = 0, localKernel = Normal(0, 1))
    nMutated[1] == 0 && return (0,[0])
    if mutKernel == "piecewise" 
        nNonLocal = rand(Binomial(nMutated[2], longJumpProb))
        return (nMutated[1], [Distributions.rand!(localKernel, zeros(nMutated[2] - nNonLocal)); longJumpLength .* ones(nNonLocal)])
    end
    error("Distribution not yet implemented or mispelled")
end

function getDisplacementPiecewise(nMutated::Tuple{Int64, Int64}, longJumpProb::Float64, longJumpLength::Int64, localKernel::Distribution{Univariate, Continuous})
    xi = rand(nMutated[2])
    nNonLocal = sum(xi .< longJumpProb)
    return (nMutated[1], [Distributions.rand!(localKernel, zeros(nMutated[2] - nNonLocal)); longJumpLength .* ones(nNonLocal)])
end

function displacementToJump(mutDisplacement, maxIdx, boundaryCond)
    out = sparsevec(zeros(maxIdx))
    jumpToIdx::Vector{Int64} = round.(mutDisplacement[2] .+ mutDisplacement[1])
    boundaryCond == 1 && (out =  sparsevec([min(max(jump, 1), maxIdx) for jump in jumpToIdx], ones(length(jumpToIdx)), maxIdx))
    boundaryCond == 2 && (out =  sparsevec([jump >= 1 ? jump <= maxIdx ? jump : max(maxIdx - jump, 1) : min(-jump + 1, maxIdx) for jump in jumpToIdx], ones(length(jumpToIdx)), maxIdx))
    boundaryCond == 3 && (out = sparsevec([min(max(jump, 1), maxIdx) for jump in jumpToIdx], [jump > 1 && jump < maxIdx ? 1 : 0 for jump in jumpToIdx], maxIdx))
    out
end

function immuneDeaths(nh, totalDeaths)
    hDeath = Distributions.rand!(DiscreteUniform(1, sum(nh)), zeros(totalDeaths))
    supressIdx::Vector{Int64} = [findfirst(cumsum(nh) .>= selected) for selected in hDeath]
    Array(sparsevec(supressIdx, ones(totalDeaths), length(nh)))
end

function initialDistribution(distType::String, sigma::Real, x)
    gaussianCond(x) = exp(-x^2/(2*sigma^2))/sqrt(2pi*sigma^2)
    skewedGaussianCond(x) = exp(-x^2/(2*sigma^2))*(1 + erf(-4x/(sqrt(2)*sigma)))

     virusIC = Vector{Int64}(undef, length(x))
     immuneIC = Vector{Int64}(undef, length(x))

    if distType == "gaussianBarrier"
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = zero(virusIC)
        immuneIC[findfirst(virusIC .> 0)] = (useFiniteMemory ? Nh*M : Nh)
    elseif distType == "skewedSteadyState"
        virusFloat = skewedGaussianCond.(x)
        virusIC = round.(virusFloat .* N0 ./ sum(virusFloat))
        immuneIC = round.(Nh/r*log(R0) .* (1 .+ erf.(.-x ./ sqrt(2*sigma^2))) ./ 2)
    elseif distType == "steadyState"
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = round.(Nh/r*log(R0) .* (1 .+ erf.(.-x ./ sqrt(2*sigma^2))) ./ 2)
    else
        virusIC = round.(N0 .* gaussianCond.(x))
        immuneIC = zero(virusIC)
    end

    return virusIC, immuneIC
end

function vNEquation(vN)
    return [vN[2] - Nh*s*vN[1], vN[1] - D^(2/3)*s^(1/3)*(24log(vN[2]*(D*s^2)^(1/3)))^(1/3)]
end

# =======================================
# Simulation parameters
# =======================================

dt = 0.1
R0 = 1.02
r = 1
Nh::Int = 1e6
theoreticalN0::Bool = 0
theoreticalN0 || (N0::Int = 3e4)
rInt::Int = ceil(r)
H(x) = exp.(-abs.(x)/r) 
Hkernel = H(-5*rInt:5*rInt)
HkernelHalfLength::Int = floor(length(Hkernel)/2)

# Mutation parameters
mutationRate = 4
mutationAv = 0
mutationScale = 5

mutationKernel = Normal(mutationAv, mutationRate)   
kernelType = string(typeof(mutationKernel))
nonLocalMutProb = 1e-6
nonLocalJump = 70

localKernel = Normal(mutationAv, mutationScale)
localKernelType = string(typeof(localKernel))
localDist = localKernelType[1: findfirst('{', localKernelType) - 1] * string(mutationScale)

if mutationKernel != "piecewise"
    dist = kernelType[1: findfirst('{', kernelType) - 1] * string(mutationScale) 
else 
    dist = mutationKernel * string(nonLocalJump) * "/" * "nonLocalProb" * string(nonLocalMutProb) * "/" * localDist
end

# Macroscopic parameters
s = log(R0)/r                       # Fitness gradient
D = mutationRate*mutationScale^2/2  # Difusion coeff


# boundaryConditions
boundaryConditions = Dict([(1, "bounded"), (2, "reflecting"), (3, "absorbing")])
bc = 3

# =================================================
# Initialisation
# =================================================

vo = D^(2/3)*s^(1/3)*(24log(Nh/1000*(D*s^2)^(1/3)))^(1/3)
No = round(Nh * vo / s)
vtheo = s
Ntheo = 1e3
e = false
try
    nlsolve(vNEquation, [vo, No]).zero
catch exc
    global e = true
end
e ? Ntheo = Int(No) : ((vtheo, Ntheo) = nlsolve(vNEquation, [vo, No]).zero)
theoreticalN0 && (N0 = Int(round(Ntheo)))
sigma = sqrt(vo / s)

xmax::Int64 = 3000
x = -Int(max(round(5*r), round(5*sigma))):xmax
maxIdx = length(x)

nx0, hx0 = initialDistribution("steadyState", sigma, x)

# Simulation and sampling
tmax = 2000
dtSampling = 1
idxSampling::Int = round(dtSampling/dt)

# Variable initialisation
nx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
nx[1,:] = nx0
hx = Array{Int64, 2}(undef, Int(round(tmax/dtSampling+1)), maxIdx)
hx[1,:] = hx0

Nt = Vector{Int64}(undef, Int(round(tmax/dt+1)))
Nt[1] = sum(nx0)
xt = Vector{Float64}(undef, Int(round(tmax/dt+1)))
xt[1] = sum(nx0.*x)/Nt[1]

# ========================================================
# Simulation
# ========================================================
println("===============START OF THE SIMULATION==============")
t = 0:dt:tmax

# Instantaneous fields
hxLoc = hx0
nxLoc = nx0

for i in 2:length(t)
    
    # Virus growth
    local c = conv(hxLoc, Hkernel)[HkernelHalfLength + 1: end - HkernelHalfLength]
    local R = R0 .* exp.(-c ./ Nh)
    nxGrowth = rand.(Poisson.(R .* nxLoc .* dt))
    nxDeath = rand.(Poisson.(nxLoc .* dt))
    global nxLoc .= max.(nxLoc .+ nxGrowth .- nxDeath, 0)
    global Nt[i] = sum(nxLoc)

    # Mutations
    nxMutated = sparsevec(rand.(Binomial.(nxLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
    mutationDisplacements = getDisplacement.(iszero(nxMutated) ? [(0, 0)] : tuple.(nxMutated.nzind, nxMutated.nzval), mutationKernel, nonLocalMutProb, nonLocalJump, localKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
    nxJump = displacementToJump.(mutationDisplacements, maxIdx, bc) # 4.643 ms
    global nxLoc = nxLoc - Array(nxMutated) + Array(sum(nxJump)) # Move mutated viruses
    
    # Immune evolution
    hxLocGrowth = rand.(Poisson.(nxLoc.*(dt)))
    global hxLoc += hxLocGrowth

    # Sampling
    xt[i] = sum(nxLoc.*x)/Nt[i]
    if i % idxSampling == 1
        nx[Int((i-1)/idxSampling + 1), :] = nxLoc
        hx[Int((i-1)/idxSampling + 1), :] = hxLoc
    end
end 

if iszero(nxLoc) 
    idxExtinguished = findfirst([iszero(nx[i,:]) for i in eachindex(nx[:,1])])
    (x[maximum([findlast(nx[i, :] .!= 0) for i in 1:idxExtinguished-1])] > xmax - 3*sigma) ? println("WARNING: virus escaped") : println("WARNING: virus extinguished")
end

# ==================================================
# Data storage
# ==================================================

NtSampled = [Nt[Int((i-1)*idxSampling + 1)] for i in 1:size(nx)[1]]
xtSampled = [xt[Int((i-1)*idxSampling + 1)] for i in 1:size(nx)[1]]
tSampled = [t[Int((i-1)*idxSampling + 1)] for i in 1:size(nx)[1]]

sampledNxtTable = Tables.table([tSampled xtSampled NtSampled], header = ["t", "xt", "Nt"])
NxtTable = Tables.table([t xt Nt], header = ["t", "xt", "Nt"])
xnxTable = Tables.table([x transpose(nx)], header = ["x"; ["t = $(t)" for t in tSampled]])
hxTable = Tables.table(transpose(hx), header = ["t = $(t)" for t in tSampled])

# boundaryCondition = boundaryConditions[bc]

dir = ("simulations/1D/" * dist * "/dt$(dt)_N0$(theoreticalN0 ? "theo" : N0)_Nh$(Nh)_R0$(R0)_r$(r)_mu$(mutationRate)_tmax$(tmax)_xSize$(xmax)")
isdir(dir) || mkpath(dir)

fileNxt = "Nxt.csv"
fileNxtSampled = "NxtSampled_dtSampling$(dtSampling).csv"
filexnx = "xnx_dtSampling$(dtSampling).csv"
filehx = "hx_dtSampling$(dtSampling).csv"

CSV.write(joinpath(dir, fileNxt), NxtTable)
CSV.write(joinpath(dir, fileNxtSampled), sampledNxtTable)
CSV.write(joinpath(dir, filexnx), xnxTable)
CSV.write(joinpath(dir, filehx), hxTable)