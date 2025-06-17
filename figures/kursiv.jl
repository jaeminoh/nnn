using CairoMakie, LaTeXStrings, NPZ

filename::String="data/kursiv.npz"
lw::Int=3

with_theme(theme_latexfonts()) do
    
    data = npzread(filename)
    uu = data["sol"]
    tt = data["tt"]


    Nx = size(uu, 2)
    xx = LinRange(0, 32 * pi, Nx)
    f = Figure(size=(500, 400), fontsize=22) # Set a global font family
    
    ax = Axis(f[1, 1],
                xlabel = L"$t$",
                ylabel = L"$x$",
                title = "Kuramoto-Sivashinsky equation",
                yticks = (0:16π:32π, ["0", L"16π", L"32π"]),
               )
    heatmap!(f[1,1], tt, xx, uu)
    Colorbar(f[1, 2])
    resize_to_layout!(f)

    save("data/kursiv.pdf", f)
    println("Done! See data/kursiv.pdf")
end