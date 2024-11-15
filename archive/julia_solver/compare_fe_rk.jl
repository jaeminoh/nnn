using CairoMakie, LaTeXStrings

include("lorenz96_solver.jl")

sol_e = solve(prob, Euler(), dt=dt; saveat = tt)
sol_r = solve(prob, Tsit5(), saveat = tt)

function compare(t1_index::Int = 61)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=18, size=(900, 300))

        e = stack(sol_e.u[begin:t1_index]; dims=1)
        ax = Axis(fig[1, 1], title="Forward Euler", xlabel=L"t", ylabel=L"x")
        heatmap!(ax, e)

        r = stack(sol_r.u[begin:t1_index]; dims=1)
        ax = Axis(fig[1, 2], title="Tsit5")
        heatmap!(ax, r)

        err = abs.(e - r)
        ax = Axis(fig[1, 3], title="Absolute Error")
        heatmap!(ax, err)

        Colorbar(fig[1, 4], limits = (0, maximum(err)))

        resize_to_layout!(fig)
        save("euler_rk_$(t1_index-1)timesteps.pdf", fig)
    end
end
