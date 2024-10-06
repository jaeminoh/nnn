using OrdinaryDiffEq
using Random
using DALS: lorenz96!

N = 128
ensemble_size = 40
tspan = (0, 5)
u0_ensemble = [8 .+ randn(N) for i in 1:ensemble_size]

function solve_ensemble(u0_ensemble)
    sol = []
    for u0 in u0_ensemble
        prob = ODEProblem(lorenz96!, u0, tspan)
        push!(sol, solve(prob, Tsit5()))
    end
    return sol
end

sol_ensemble = solve_ensemble(u0_ensemble)

# visualization
function draw(sol)
    heatmap(stack(sol))
end