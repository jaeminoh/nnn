using Plots
using NPZ
using LaTeXStrings

function animate(fname::String="figures/kursiv.mp4"; noise::Int=50, scale::Number=1)
    scalefontsizes()
    default(fontfamily="Computer Modern",
        linewidth=3, framestyle=:box, grid=false,
        size=(1200, 400), ylims=(-3.5,3.5), dpi=300, label=nothing,
        legend=:outertop, legend_columns=-1)
    scalefontsizes(scale)
    result = npzread("Kursiv/results/kursiv_lr0.001_epoch200_noise$(noise)_rank256_test.npz")
    tt = result["tt"]
    uu = result["uu"]
    xx = LinRange(0, 32 * π, 129)[begin:end-1]
    ff = result["uu_f"]
    yy = result["yy"]
    aa = result["uu_a"]

    # animate
    l = @layout [a b c]

    p1 = plot(xx, uu[1, :], label="Reference")
    plot!(p1, xx, ff[1, :], label="Forward Euler", linestyle=:dash)

    p2 = plot(xx, uu[1, :], label="Reference")
    plot!(p2, xx, yy[1, :], label="Observation", linestyle=:dash)

    p3 = plot(xx, uu[1, :], label="Reference")
    plot!(p3, xx, aa[1, :], label="Filtered", linestyle=:dash)

    plot(p1, p2, p3, layout=l, suptitle="Time: $(round(tt[1], digits=4))")

    tt = tt[2:end]
    ii = 4:2:length(tt)
    anim = @animate for i in ii

        p1 = plot(xx, uu[i, :], label="Reference")
        plot!(p1, xx, ff[i, :], label="Forward Euler", linestyle=:dash)

        p2 = plot(xx, uu[i, :], label="Reference")
        plot!(p2, xx, yy[i, :], label="Observation", linestyle=:dash)

        p3 = plot(xx, uu[i, :], label="Reference")
        plot!(p3, xx, aa[i, :], label="Filtered", linestyle=:dash)

        plot(p1, p2, p3, layout=l,
            suptitle="Kuramoto-Sivashinsky Equation. Time: $(round(tt[i], digits=4))")
    end
    mp4(anim, fname, fps=50)
    println("Done!")
end

function draw(noise::Int=50)
    animate(noise=noise, scale=1.2)
end