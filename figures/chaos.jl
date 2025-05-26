using CairoMakie, LaTeXStrings, NPZ

filename::String="data/chaos.npz"
lw::Int=3

with_theme(theme_latexfonts()) do
    
    data = npzread(filename)
    uu = data["uu"]
    vv = data["vv"]

    u0 = uu[1, :]
    u1 = uu[2, :]
    v0 = vv[1, :]
    v1 = vv[2, :]

    Nx = length(u0)
    x_values = 1:Nx
    f = Figure(size=(800, 400), fontsize=22) # Set a global font family
    
    
    # Subplot for t=0
    ax0 = Axis(f[1, 1],
                xlabel = L"$x$",
                title = L"$t=0$")
    
    lines!(ax0, x_values, u0, label=L"$u$", linewidth=lw)
    lines!(ax0, x_values, v0, label=L"$v$", linestyle=:dash, linewidth=lw, color=:red)
    #axislegend(ax0, position=:rt)

    # Subplot for t=1
    ax1 = Axis(f[1, 2],
                xlabel = L"$x$",
                title = L"$t=1$")
    
    lines!(ax1, x_values, u1, linewidth=lw)
    lines!(ax1, x_values, v1, linestyle=:dash, linewidth=lw, color=:red)

    Legend(f[1, 3], ax0, orientation=:vertical)

    resize_to_layout!(f)
    
    save("data/chaos.pdf", f)
    println("Done! See data/chaos.pdf")
end