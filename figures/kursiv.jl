using CairoMakie, LaTeXStrings, NPZ


lw::Int=3

with_theme(theme_latexfonts()) do
    f = Figure(size=(800, 400), fontsize=22)
    

    data_nonlinear = npzread("data/kursiv_nonlinear.npz")
    data_linear = npzread("data/kursiv_linear.npz")
    ts = data_linear["tt"] # time points
    ys = data_linear["yy"] # observations
    us = data_linear["uu"] # reference solution
    Nx = size(us, 2)
    xs = LinRange(0, 32 * pi, Nx)

    us_nonlinear = data_nonlinear["uu_a"] # NNN
    us_linear = data_linear["uu_a"] # linear nudging
    

    
    ax1 = Axis(f[1, 1],
                xlabel = L"x",
                xticks = (0:16π:32π, ["0", L"16π", L"32π"]),
               )
    lines!(ax1, xs, us[end, :], linewidth=lw, color=:black, label="Reference")
    lines!(ax1, xs, us_nonlinear[end, :], linewidth=lw, linestyle=:dashdot, label="NNN")
    lines!(ax1, xs, us_linear[end, :], linewidth=lw, color=:red, linestyle=:dash, label="Linear")
    Legend(f[0, 1], ax1, orientation=:horizontal)


    ax2 = Axis(f[2, 1],
                xlabel = L"t",
                ylabel = L"$\Vert u - \hat{u} \Vert$",
               )
    es_nonlinear = sqrt.(dropdims(sum((us_nonlinear .- us).^2, dims=2), dims=2) / size(us, 2))
    es_linear = sqrt.(dropdims(sum((us_linear .- us).^2, dims=2), dims=2) / size(us, 2))
    lines!(ax2, ts[2:end], es_nonlinear, linewidth=lw, label="NNN", linestyle=:dash)
    lines!(ax2, ts[2:end], es_linear, linewidth=lw, color=:red, linestyle=:dashdot, label="Linear")

    
    data = npzread("data/kursiv.npz")
    us = data["sol"]
    ts = data["tt"]

    ax3 = Axis(f[:, 2],
                xlabel = L"$t$",
                ylabel = L"$x$",
                yticks = (0:16π:32π, ["0", L"16π", L"32π"]),
               )
    heatmap!(ax3, ts, xs, us)
    Colorbar(f[:, 3])

    

    resize_to_layout!(f)

    save("data/kursiv.pdf", f)
    println("Done! See data/kursiv.pdf")
end