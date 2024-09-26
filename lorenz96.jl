using OrdinaryDiffEq, StaticArrays

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 200.0)

function lorenz96!(du, u, p, t)
    N = length(du)
    for n ∈ 1:N
        du[n] = (u[mod1(n+1, N)] - u[mod1(n-2, N)]) * u[mod1(n-1, N)] - u[n] + 8.0
    end
end

prob = ODEProblem(lorenz96!, u0, tspan)
sol = solve(prob, Tsit5())

using Plots, LaTeXStrings
u_img = stack(sol.u, dims=2)
heatmap(u_img, xaxis=L"t", yaxis=L"x", title="Lorenz 96", dpi=300)
savefig("figures/lorenz96.pdf")
