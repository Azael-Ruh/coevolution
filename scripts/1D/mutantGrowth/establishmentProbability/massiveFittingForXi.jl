include("../../../../code/mutantGrowth/secondMutantStudy.jl")

zetaGrid = 0.10:0.01:0.2
dGrid = 10 .^(-2:0.1:-0.5)
dr = 1e-3
xiGrid = zeros(3, length(zetaGrid), length(dGrid))

xi0 = 2.338
xi1 = 1.019

for i in eachindex(zetaGrid), j in eachindex(dGrid)

    println("=========== Calculation #$((i-1)*length(zetaGrid) + j) =========")

    zeta = zetaGrid[i]
    d = dGrid[j]
    vs = sqrt(zeta*4*d^3)
    Ds = d^3

    (rFixed, xiGrid[1,i,j], rM, xiGrid[2,i,j]) = getMatchingLimits(vs, Ds)
    rc0 = (rFixed + rM)/2

    println("Calculating point zeta = $zeta, d = $d, vs = $vs, Ds = $Ds")
    
    rVect = -10*max(zeta,d):dr:2.5*(zeta + 2.338d)
    xic = NaN
    wNumeric = zero(rVect)
    try
        wNumeric = getNumericalFixationProbability(rVect, dr, vs, Ds, rc0)
        println("Success for numerical resolution of the ODE")
        try
            rc, xic = getBestMatchingPoint(wNumeric, vs, Ds, rVect, rc0)
            println("Success for fitting rc")
        catch e
            println("Error for fitting rc")
            println("Error: $e")
        end
    catch e
        println("Error for numerical resolution")
        println("Error: $e")
    end

    xiGrid[3, i, j] = xic
end

chiVect = [z/d for z in zetaGrid, d in dGrid][:]
chiSynt = 0:0.1:maximum(chiVect)

# Fitting for xic(chi)

I = sortperm([z/d for z in zetaGrid, d in dGrid][:])
chi = [z/d for z in zetaGrid, d in dGrid][:][I]
xic = xiGrid[3,:,:][:][I]

function alpha(r, chi)
    zeta = 0.1
    d = zeta / chi
    vs = sqrt(zeta *4 * d^3)
    Ds = d^3
    return vs^2/2Ds - vs/Ds^(1/3)*airyaiprime((vs^2/4Ds - r)/Ds^(1/3))/airyai((vs^2/4Ds - r)/Ds^(1/3))
end

function functionToZero(r, p)
    chi = p[1]
    theta = p[2]
    return chi./(chi .+ theta) .* r .- alpha.(r, chi)
end

function functionToFit(chiVect, p)
    theta = p[1]
    xic = zero(chiVect)
    for i in eachindex(chiVect)
        zeta = 0.1
        d = zeta / chiVect[i]
        xic[i] = (nlsolve(r -> functionToZero(r, [chiVect[i], theta]), [zeta+d]).zero[1] - zeta)/d
    end

    return xic
end

fit = curve_fit(functionToFit, chi, xic, [0.75])
fitParams = fit.param

# Second derivative condition

function secondDerivativeCond(r, p)
    zeta = p[1]
    d = p[2]
    vs = sqrt(zeta *4 * d^3)
    Ds = d^3

    alphaVect = alphac.(r, vs, Ds)
    return (alphaVect .- r)./Ds - alphaVect .*(2alphaVect .- r) ./ vs^2 .- (1 .+ alphaVect)./(1 .+ r)./vs
end

function xiSecondDerivative(zeta, d)
    return xic = (nlsolve(r -> secondDerivativeCond(r, [zeta, d]), [zeta+d]).zero[1] - zeta)/d
end

xiSecondD = [rcSecondDerivative(zeta, d) for zeta in zetaGrid, d in dGrid][:]

# Plots

plotConfig()
p = scatter(chiVect,(xiGrid[1,:,:])[:], label = raw"$\xi_F$", c = :steelblue, msc = :steelblue)
scatter!(chiVect,(xiGrid[3,:,:])[:], label = raw"Best $\xi_c$", c = :seagreen, msc = :seagreen)
scatter!(chiVect,(xiGrid[2,:,:])[:], label = raw"$\xi_M$", c = :darkorange, msc = :darkorange)
scatter!(p, chiVect, xiSecondD, label = raw"Second derivative continuity", c = :lightpink, msc = :lightpink)
hline!([1.019, 2.338], colour = :black, ls = :dash, label = raw"$\xi_0,\xi_1$")
plot!(chiSynt, xi0 .- sqrt.(1 ./chiSynt), c = :black, label = raw"High $\zeta/d$ expansions")
plot!(chiSynt, xi0 .- 2sqrt.(1 ./chiSynt), c = :black, label = "")
plot!(legend_position = :bottomright, ylims = (1, 2.35), xlabel = raw"$\zeta/d$", ylabel = raw"$\xi$")
plot!(chiSynt, (xi1 .+ sqrt.(chiSynt) ./ 2) ./ (1 .+ 1 ./ 2chiSynt), c = :black, ls = :dashdot, label = raw"Low $\zeta/d$ expansions")
plot!(chiSynt, (xi1 .+ sqrt.(chiSynt)), c = :black, ls = :dashdot, label = "")
plot!(p, chi[1:findlast(functionToFit(chi, fitParams) .< 2.1)], functionToFit(chi, fitParams)[1:findlast(functionToFit(chi, fitParams) .< 2.1)], c = :red, label = raw"Best fit for $\xi_c(\zeta/d)$")

# Error minimisation (not quite right actually :c)

function integralCondition(rc, rVect, p)
    zeta = p[1]
    d = p[2]

    wtheo = wAsymptotic.(rVect, rc, vs, Ds)
    return sum(rVect .* wtheo .- (1 .+ rVect) .* wtheo .^2)  - vs
end

function exponentialCondition(rc, rVect, p)
    zeta = p[1]
    d = p[2]

    wtheo = wAsymptotic.(rVect, rc, vs, Ds)
    return sum(exp.( vs .* (rc .- rVect) ./ Ds) .*( rVect .* wtheo .- (1 .+ rVect) .* wtheo .^2))
end

function xiIntegralConition(zeta, d)
    xiInt = xi1:0.005:xi0
    rcInt = zeta .+ d.*xiInt
    funToSolve = r -> integralCondition(r, rcInt, [zeta, d])
    return xic = (nlsolve(funToSolve, [zeta+d]).zero[1] - zeta)/d
end