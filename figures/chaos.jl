using CairoMakie, LaTeXStrings, NPZ

filename::String="chaos.npz"
lw::Int=3

with_theme(theme_latexfonts()) do
    
    data = npzread(filename)
    us = data["us"]
    vs = data["vs"]
    es = data["error"]

    u0 = us[1, :]
    u1 = us[2, :]
    v0 = vs[1, :]
    v1 = vs[2, :]

    Nx = length(u0)
    x_values = 1:Nx
    f = Figure(size=(1000, 300), fontsize=22) # Set a global font family
    
    
    # Subplot for t=0
    ax0 = Axis(f[1, 1],
                xlabel = L"$x$",
                title = L"$t=0$")
    
    lines!(ax0, x_values, u0, label=L"$u$", linewidth=lw)
    lines!(ax0, x_values, v0, label=L"$û$", linestyle=:dash, linewidth=lw, color=:red)

    # Subplot for t=1
    ax1 = Axis(f[1, 2],
                xlabel = L"$x$",
                title = L"$t=1$")
    
    lines!(ax1, x_values, u1, linewidth=lw)
    lines!(ax1, x_values, v1, linestyle=:dash, linewidth=lw, color=:red)

    Legend(f[1, 0], ax0, orientation=:vertical)

    # Subplot for error over time
    ax2 = Axis(f[1, 3],
                xlabel = L"$t$",
                ylabel = L"$\Vert u - û \Vert$",
                title = "Error over time",)

    ts = LinRange(0, 1, first(size(es)))
    lines!(ax2, ts, es, linewidth=lw, color=:black)

    resize_to_layout!(f)
    
    save("data/chaos.pdf", f)
    println("Done! See data/chaos.pdf")
end