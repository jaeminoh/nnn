using CairoMakie, LaTeXStrings, NPZ


lw::Int=3

us = npzread("data/kolmogorov_nonlinear.npz")["uu"]
us_nonlinear = npzread("data/kolmogorov_nonlinear.npz")["uu_a"]
us_linear = npzread("data/kolmogorov_linear.npz")["uu_a"]
ts = npzread("data/kolmogorov_nonlinear.npz")["tt"]

with_theme(theme_latexfonts()) do
    f = Figure(size=(1200, 400), fontsize=22)

    g1 = f[1, 1] = GridLayout()
    g2 = f[1, 2] = GridLayout()

    # Snapshot
    u = us[end, :, :]
    u_nonlinear = us_nonlinear[end, :, :]
    u_linear = us_linear[end, :, :]
    xs = LinRange(0, 2π, size(u, 2))

    limits = (minimum(stack([u, u_nonlinear, u_linear])), maximum(stack([u, u_nonlinear, u_linear])))

    ax0 = Axis(g1[1, 1], title="Ground truth", xlabel=L"x", ylabel=L"y", xticks=(0:π:2π, ["0", L"π", L"2π"]), yticks=(0:π:2π, ["0", L"π", L"2π"]))
    heatmap!(ax0, xs, xs, u, colormap=:jet, colorrange=limits)

    ax1 = Axis(g1[1, 2], title="NNN", xlabel=L"x", xticks=(0:π:2π, ["", L"π", L"2π"]))
    heatmap!(ax1, xs, xs, u_nonlinear, colormap=:jet, colorrange=limits)
    hideydecorations!(ax1)

    ax2 = Axis(g1[1, 3], title="Linear", xlabel=L"x", xticks=(0:π:2π, ["", L"π", L"2π"]))
    heatmap!(ax2, xs, xs, u_linear, colormap=:jet, colorrange=limits)
    hideydecorations!(ax2)

    Colorbar(g1[1, 4], colormap=:jet, limits=limits)


    # Error plot
    ax = Axis(g2[1, 1], xlabel=L"t", ylabel="RMSE")
    es_nonlinear = dropdims(sum((us_nonlinear .- us).^ 2, dims=(2, 3)), dims=(2,3)) / size(us, 2)^2 .|> sqrt
    es_linear = dropdims(sum((us_linear .- us).^ 2, dims=(2, 3)), dims=(2,3)) / size(us, 2)^2 .|> sqrt

    lines!(ax, ts[2:end], es_nonlinear, linewidth=lw, label="NNN", linestyle=:dash)
    lines!(ax, ts[2:end], es_linear, linewidth=lw, color=:red, linestyle=:dot, label="Linear")
    axislegend(ax, position=:rt)

    Label(g1[1, 1, TopLeft()], "(A)", font = :bold, padding = (0, 5, 5, 0), halign=:right)
    Label(g2[1, 1, TopLeft()], "(B)", font = :bold, padding = (0, 5, 5, 0), halign=:right)
    
    colsize!(f.layout, 1, Auto(2))
    resize_to_layout!(f)
    save("data/kolmogorov.pdf", f)
end

println("Done! See data/kolmogorov.pdf")