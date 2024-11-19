using Plots
using NPZ
using LaTeXStrings

function animate(type::String; noise::Int=50, scale::Number=1)
    scalefontsizes(scale)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
        linewidth=3, framestyle=:box, label=nothing, grid=false)
    scalefontsizes(scale)
    result = npzread("Kursiv/results/kursiv_lr0.001_epoch200_noise$(noise)_rank256_test.npz")
    tt = result["tt"]
    uu = result["uu"]
    xx = LinRange(0, 32 * π, 129)[begin:end-1]
    # which result?
    if type == "baseline"
        ff = result["uu_f"]
        fname = "figures/kursiv_base.mp4"
        label = "Forward Euler"
    elseif type == "observation"
        ff = result["yy"]
        fname = "figures/kursiv_obs.mp4"
        label = "Observation"
    else
        ff = result["uu_a"]
        fname = "figures/kursiv_assim.mp4"
        label = "Assimilation"
    end
    # animate
    plot(xx, uu[1, :], label="Reference", lw=3,
    legend=:outertop, legend_columns=-1, xlabel=L"x", ylabel=L"u",
    ylims=(-4.5, 4.5), title="Time: $(tt[1])")
    plot!(xx, ff[1, :], label=label, lw=3, linestyle=:dash)
    anim = @animate for (i, t) ∈ enumerate(tt[2:end])
        if i % 4 == 0
            plot(xx, uu[i, :], label="Reference", lw=3,
                legend=:outertop, legend_columns=-1, xlabel=L"x", ylabel=L"u",
                ylims=(-4.5, 4.5), title="Time: $(t)")
            plot!(xx, ff[i, :], label=label, lw=3, linestyle=:dash)
        end
    end
    mp4(anim, fname, fps=50)
    println("Done!")
end

function draw(noise::Int=50)
    for f in ["baseline", "observation", "assimilation"]
        animate(f, noise=noise, scale=1.2)
    end
end