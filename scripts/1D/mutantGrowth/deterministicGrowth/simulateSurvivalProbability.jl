
using SpecialFunctions, Plots
include("../../../../code/simulate/coevolution1DSimulationTools.jl")

function simulateMutantSurvival(v, r, s, Nh, mutationRate, mutationKernel, x, tmax, dt, x0, M0, initialCond; dx = 1)

    maxIdx = length(x)

    # Mutations
    mutationScale = std(mutationKernel)
    D = mutationRate * mutationScale^2 / 2

    # Immunity
    H(x) = exp.(-abs.(x)/r)
    if r == 0
        Hkernel = [1]
    else
        Hkernel = H(-5*ceil(r):5*ceil(r))
    end
    HkernelHalfLength::Int = floor(length(Hkernel)/2)

    t = 0:dt:tmax

    m0 = round.(Int, M0 .* initialCond(x, x0))
    mLoc::Vector{Int64} = m0
    hLoc = zero(mLoc)
    mutantExtinct = false
    tExtinct = NaN
    for i in 1:length(t)-1

        # Growth and death
        RLoc = 1 .+ s.*(x .- v.*t[i] )
        mGrowth = rand.(Poisson.(RLoc .* mLoc .* dt))
        mDeath = rand.(Poisson.(mLoc .* dt)) # rand.(Binomial.(nxLoc, 1 - exp(-dt)))
        mLoc = max.(mLoc .+ mGrowth .- mDeath, 0)

        mutantExtinct = iszero(mLoc)
        if mutantExtinct
            tExtinct = t[i+1]
            println("Mutant extinct")
            break
        end

        # Mutations
        mMutated = sparsevec(rand.(Binomial.(mLoc, 1 - exp(-mutationRate*dt)))) # 96.2 μs
        mutationDisplacements = getDisplacement.(iszero(mMutated) ? [(0, 0)] : tuple.(mMutated.nzind, mMutated.nzval), mutationKernel) # 267.5 μs (~10 mut per x), 511.135 μs (~100 mut per x), 32.847 ms (~ 1000 mut per x)
        mJump = displacementToJump.(mutationDisplacements, maxIdx, dx = dx) # 4.643 ms
        mLoc = mLoc - Array(mMutated) + Array(sum(mJump)) # Move mutated viruses
    end

    return mutantExtinct, tExtinct
end

v = 0.1
s = 2e-3
r = 40
Nh = 1000000
dx = 1
x = -40:dx:120

# Mutations
mutationRate = 0.2
mutationKernel = 2*Bernoulli(0.5) - 1
mutationScale = std(mutationKernel)
D = mutationRate * mutationScale^2 / 2

tmax = 150
dt = 0.001

x0Array = 1:50
M0 = 1 # ceil(Int, 1/(x0*s))
v0 = 10 # 2*D/(x0*s)
gaussianCond(x,x0,var) = exp.(-(x.-x0).^2 ./ 2var) ./ sum(exp.(-(x.-x0).^2 ./ 2var))
deltaCond(x,x0) = 1 .* (x .== x0)
initialCondition = deltaCond #(x,x0) -> gaussianCond(x, x0, v0)


nRuns = 1000
survProb = Vector{Float64}(undef, length(x0Array))

for i in eachindex(x0Array)
    
    x0 = x0Array[i]
    println("========================== Position x0 = $x0 ==========================")
    
    t = 0:dt:tmax
    numSurvived = 0
    
    for run in 1:nRuns
        println("Run $run started")
        mutantExtinct,  = simulateMutantSurvival(v, r, s, Nh, mutationRate, mutationKernel, x, tmax, dt,x0, M0, initialCondition, dx = dx)
        println("Run $run finished")
        numSurvived += 1-mutantExtinct
    end
    survProb[i] = numSurvived/nRuns

end

# Fitness space paremeters

zeta = v^2/4D
vs = v * s
Ds = D * s^2
d = Ds^(1/3)

# Numerical solution
function discretisedODE!(f, r, w, dr)
    f[1] = w[1]
    f[end] = w[end] - r[end]/(1 + r[end])
    f[2:end-1] .= vs/2dr .* (w[3:end] - w[1:end-2]) .- r[2:end-1] .* w[2:end-1] .+ (1 .+ r[2:end-1]) .* w[2:end-1].^2 .- Ds/dr^2 .* (w[3:end] .- 2 .* w[2:end-1] .+ w[1:end-2])
end

uT = v^2/(4D*s)+2.3381*(D/s)^(1/3)-2D/v
dr = 5e-4
rT = s*uT
rVect = -10rT:dr:10rT

alphac(rc) = 2zeta - vs/d*airyaiprime((zeta - rc)/d)/airyai((zeta - rc)/d)
wc(rc) = (rc-alphac(rc))/(1+rc)
Ac(rc) = wc(rc)/(exp(vs*rc/(2Ds))*airyai((zeta - rc)/d))
Cc(rc) = vs * exp(rc^2/2vs)/wc(rc) - (vs * exp(rc^2/2vs) + sqrt(pi * vs / 2) * erfi(rc / sqrt(2vs)))
whighGood(r, rc) = vs * exp(r^2 / 2vs) / (vs * exp(r^2 /2vs) + Cc(rc) + sqrt(pi * vs / 2) * erfi(r / sqrt(2vs)))
wlowGood(r, rc) = Ac(rc) * airyai((zeta-r)/d) * exp(vs*r/2Ds)
wtheo(r, rc) = whighGood.(r,rc) .* (r .> rc) .+ wlowGood.(r,rc) .* (r .< rc)
rc0 = zeta + 1.6d
w0 = wtheo(rVect, rc0)
w0[1] = 0

# Solution
discretisedODEToSolve! = (f,w) -> discretisedODE!(f, rVect, w, dr)
sol = nlsolve(discretisedODEToSolve!, w0)
wSol = sol.zero

# Plot
plotConfig()
sigma = sqrt.((survProb.*nRuns .+ 1) .* (nRuns .- survProb.*nRuns .+1) ./ ((nRuns .+ 2).^2 .* (nRuns .+ 3)))
sigma = [max(sigma[i], 1 / nRuns) for i in eachindex(sigma)]
scatter(x0Array./uT, survProb, yerr = sigma, c = :black, ms = 3, label = "Simulation data", xlabel = raw"$x/u_T$", ylabel = raw"$w(x)$")
plot!(rVect./rT, wSol, xlims = (0, maximum(x0Array./uT)), c = :coral, lw = 1.5, ylims = (0,0.12), label = "Numerical solution of the branching process")
plot!(x0Array./uT, s.*x0Array, c = :black, ls = :dash, lw = 1, label = "Haldane limit")