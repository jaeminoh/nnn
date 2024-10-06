using OrdinaryDiffEq
using LaTeXStrings, CairoMakie
using DALS: lorenz96!

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 200.0)

prob = ODEProblem(lorenz96!, u0, tspan)
sol = solve(prob, Tsit5())

function draw_and_save(t1)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=18, size=(800, 200))
        ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"x")
        
        tt = LinRange(0, t1, N * 4)
        mat = stack(sol.(tt), dims=1)
        
        CairoMakie.heatmap!(ax, tt, 1:N, mat)

        resize_to_layout!(fig)
        save("figures/lorenz96_t$(t1).pdf", fig)
    end
end

draw_and_save(5)
draw_and_save(200)
