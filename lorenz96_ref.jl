using OrdinaryDiffEq
using Plots, LaTeXStrings
using LatentDA: lorenz96!

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 200.0)

prob = ODEProblem(lorenz96!, u0, tspan)
sol = solve(prob, Tsit5())

u_img = stack(sol.u, dims=2)
heatmap(u_img, xaxis=L"t", yaxis=L"x", title="Lorenz 96", dpi=300)
savefig("figures/lorenz96.pdf")
