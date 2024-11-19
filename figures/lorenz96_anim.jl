using Plots
using NPZ
using LaTeXStrings

function animate(type::String; noise::Int=50, scale::Number=1)
    scalefontsizes()
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
        linewidth=3, framestyle=:box, label=nothing, grid=false)
    scalefontsizes(scale)
    result = npzread("Lorenz96/results/lorenz_lr0.001_epoch200_noise$(noise)_rank256_test.npz")
    tt = result["tt"]
    uu = result["uu"]
    # which result?
    if type == "baseline"
        ff = result["uu_f"]
        fname = "figures/lorenz_base.mp4"
        label = "Forward Euler"
    elseif type == "observation"
        ff = result["yy"]
        fname = "figures/lorenz_obs.mp4"
        label = "Observation"
    else
        ff = result["uu_a"]
        fname = "figures/lorenz_assim.mp4"
        label = "Assimilation"
    end
    # animate
    plot(uu[1, :], label="Reference", lw=3,
    legend=:outertop, legend_columns=-1, xlabel=L"x", ylabel=L"u",
    ylims=(-10, 15), title="Time: $(tt[1])")
    plot!(ff[1, :], label=label, lw=3, linestyle=:dash)
    tt = tt[2:end]
    ii = 10:10:length(tt)
    anim = @animate for i in ii
        plot(uu[i, :], label="Reference", lw=3,
            legend=:outertop, legend_columns=-1, xlabel=L"x", ylabel=L"u",
            ylims=(-10, 15), title="Time: $(round(tt[i], digits=4))")
        plot!(ff[i, :], label=label, lw=3, linestyle=:dash)
    end
    mp4(anim, fname, fps=15)
    println("Done!")
end

function draw(noise::Int=100)
    for f in ["baseline", "observation", "assimilation"]
        animate(f, noise=noise, scale=1.2)
    end
end