using JLD2, CairoMakie

r = 40
R0 = 1.2
tmax = 1000.
totalRuns = 400

saveDir = "simulations/jumpEffects/"

vars = load(joinpath(saveDir, "jumpEffectsSpeed_r$(r)R0$(R0)tmax$(tmax)totalRuns$(totalRuns).jld2"))
mutationRateVect = vars["mutationRateVect"]
nonLocalJumpVect = vars["nonLocalJumpVect"]
newVMat = vars["newVMat"]

vars = load(joinpath(saveDir, "jumpEffectsSProb_r$(r)R0$(R0)tmax$(tmax)totalRuns$(totalRuns).jld2"))
survivalProb = vars["survivalProb"]

set_theme!(theme_latexfonts())
update_theme!(fontsize=15)
fig = Figure(size=(700, 400))
ax1 = Axis(fig[1, 1], ylabel=L"Overall mutation rate $\mu$", xticklabelsize=12, yticklabelsize=12)
hm1 = heatmap!(ax1, nonLocalJumpVect, mutationRateVect, newVMat, colormap=:magma)
Colorbar(fig[1, end+1], hm1, label=L"Average speed $v$")

ax2 = Axis(fig[1, end+1], xlabel=L"Non-local jump size $\Delta$", xticklabelsize=12, yticklabelsize=12)
hm2 = heatmap!(ax2, nonLocalJumpVect, mutationRateVect, survivalProb, colormap=cgrad([:white, cgrad(:blues, 3)[1], cgrad(:blues, 3)[2], cgrad(:blues, 3)[3]]))
Colorbar(fig[1, end+1], hm2, label="Survival probability")

linkyaxes!(ax1, ax2)

colors = cgrad(:magma)[range(0, 1, length=27)]
ax3 = Axis(fig[2, 1], xlabel=L"Non-local jump size $\Delta$", ylabel=L"Average speed $v$", xticklabelsize=12, yticklabelsize=12, xgridvisible=false, ygridvisible=false)
# linkxaxes!(ax1, ax3)
[lines!(ax3, nonLocalJumpVect, newVMat[:, i], color=colors[i]) for i in eachindex(mutationRateVect)]
Colorbar(fig[2, 2], limits=extrema(mutationRateVect), colormap=:magma, label=L"Overall mutation rate $\mu$")