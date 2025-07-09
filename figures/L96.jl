using CairoMakie, LaTeXStrings, NPZ


lw::Int=3

with_theme(theme_latexfonts()) do
    f = Figure(size=(800, 400), fontsize=22)

    # solution profile
    data = npzread("data/L96.npz")
    us = data["sol"]
    ts = data["tt"]
    Nx = size(us, 2)
    xs = 1:Nx

    ax1 = Axis(f[1:2, 1],
                xlabel = L"$t$",
                ylabel = L"$x$",
                )
    heatmap!(ax1, ts, xs, us, colormap=:jet)
    Colorbar(f[0, 1], limits=(minimum(us), maximum(us)), colormap=:jet, vertical=false)

    # Comparision: linear vs. nonlinear
    data_nonlinear = npzread("data/L96_nonlinear.npz")
    data_linear = npzread("data/L96_linear.npz")
    ts = data_linear["tt"] # time points

    # snapshot at the last time point
    us = data_linear["uu"] # reference solution
    us_nonlinear = data_nonlinear["uu_a"] # NNN
    us_linear = data_linear["uu_a"] # linear nudging
    
    ax2 = Axis(f[1, 2],
                xlabel = L"x",
                ylabel = L"u",
               )
    lines!(ax2, xs, us[end, :], linewidth=lw, color=:black, label="GT")
    lines!(ax2, xs, us_nonlinear[end, :], linewidth=lw, linestyle=:dash, label="NNN")
    lines!(ax2, xs, us_linear[end, :], linewidth=lw, color=:red, linestyle=:dot, label="Linear")
    Legend(f[0, 2], ax2, orientation=:horizontal)

    # error plot
    ax3 = Axis(f[2, 2],
                xlabel = L"t",
                ylabel = "RMSE",
                xticks = (255:60:315, ["255", "315"]),
               )
    es_nonlinear = dropdims(sum((us_nonlinear .- us).^2, dims=2), dims=2) / size(us, 2) .|> sqrt
    es_linear = dropdims(sum((us_linear .- us).^2, dims=2), dims=2) / size(us, 2) .|> sqrt
    lines!(ax3, ts[2:end], es_nonlinear, linewidth=lw, label="NNN", linestyle=:dash)
    lines!(ax3, ts[2:end], es_linear, linewidth=lw, color=:red, linestyle=:dot, label="Linear")

    resize_to_layout!(f)

    save("data/L96.pdf", f)
    println("Done! See data/L96.pdf")
end