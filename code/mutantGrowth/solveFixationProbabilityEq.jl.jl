using Plots, SpecialFunctions

# Model settings
xi = 0.1
zeta = xi/4
d = 0.001
D = d^3
v = sqrt(xi*d^3)

# Differential equation
function survivalProbEq(y, x)
    w = y[0]
    dw = y[1]
    
    return [dw, (v*dw - x*w + (1 + x)*w^2)/D]
end

dr = d/1e4
rArr = 0:dr:xi/4+3*d
w = Vector{Float64}(undef, length(rArr))

rc = zeta
w[1] = zeta/(1+zeta)*exp(v*(rArr[1]-rc)/(2D))*airyai((zeta - rArr[1])/d)/airyai((zeta - rc)/d)
w[2] = zeta/(1+zeta)*exp(v*(rArr[2]-rc)/(2D))*airyai((zeta - rArr[2])/d)/airyai((zeta - rc)/d)
w[2] = w[1] + (w[2]-w[1])/2^(9.9525)

for i in eachindex(rArr)[2:end-1]
    w[i+1] = (w[i]*rArr[i] + v*w[i-1]/2dr-(1+rArr[i])*w[i]^2+D*(w[i-1]-2w[i])/dr^2)/((v/2dr)-D/dr^2)
end
