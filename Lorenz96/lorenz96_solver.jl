using OrdinaryDiffEq, LaTeXStrings, CairoMakie
using NPZ
using DALS: lorenz96!

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 200.0)
dt = 0.01
tt = tspan[1]:dt:tspan[2]

prob = ODEProblem(lorenz96!, u0, tspan)

function draw_and_save(solver::String)
    if solver == "Euler"
        sol = solve(prob, Euler(), dt=dt; saveat=tt)
    elseif solver == "Tsit"
        sol = solve(prob, Tsit5(); saveat=tt)
    end
    mat = stack(sol.u; dims=1)

    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=18, size=(800, 200))
        ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"x")
        npzwrite("Lorenz96/data/$(solver).npz", Dict("tt" => tt, "sol" => mat))
        mat = mat[1:50:end, :]
        CairoMakie.heatmap!(ax, tt[1:50:end], 1:N, mat)

        resize_to_layout!(fig)
        save("figures/lorenz96_$(solver).pdf", fig)
    end
    println(solver, " Done!")
end

draw_and_save("Euler")
draw_and_save("Tsit")
