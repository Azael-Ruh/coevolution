using SpecialFunctions

function mutantUpdateRule!(dmdt, x, t, mVect, params)
    v = params[1]
    D = params[2]
    s = params[3]
    dx = params[4]
    dmdt[1] = 0
    dmdt[end] = 0
    dmdt[2:end-1] .= s .* (x[2:end-1] .- v * t) .* mVect[2:end-1] + D .* (mVect[1:end-2] - 2mVect[2:end-1] + mVect[3:end])./dx^2
    dmdt
end

v = 0.1
D = 0.1
s = 1e-3
dx = 0.5
x = -25:dx:300

tmax = 1000
dt = 0.004
dtSampling = 0.1
t = 0:dt:tmax
tSampling = 0:dtSampling:tmax

x0 = 60
M0 = 1/(x0*s)
gaussianCond(x,x0,var) = sqrt(1/(2 .* pi .* var)) .* exp.(-(x.-x0).^2 ./ var)
deltaCond(x,x0) = x .== x0 
m0 = M0 .* gaussianCond(x, x0, 2 * D / (x0 *s))
m0[x .== x0] = m0[x .== x0] + M0 - sum(m0)

# m = Array{Float64}(undef, length(tSampling), length(x))
# m[1,:] = m0
M = Vector{Float64}(undef, length(t))
M[1] = M0
xm = Vector{Float64}(undef, length(t))
xm[1] = sum(x .* m0) ./ M0
sigmam = Vector{Float64}(undef, length(t))
sigmam[1] = sum(x.^2 .* m0) ./ M0 - xm[1].^2
params = (v, D, s, dx)

dmdt = zero(m0)
mLoc = m0
for i in 1:length(t)-1
    global mLoc = mLoc + mutantUpdateRule!(dmdt, x, t[i], mLoc, params) .* dt
    M[i+1] = sum(mLoc)
    xm[i+1] = sum(x .* mLoc) ./ M[i+1]
    sigmam[i+1] = sum(x.^2 .* mLoc) ./ M[i+1] - xm[i+1].^2

    # if (t[i] % dtSampling == 0)
    #     m[Int(t[i] ./ dtSampling) + 1, :] = mLoc
    # end
    # println("i = $i")
end