using OrdinaryDiffEq, Random, NPZ
using DALS: lorenz96!

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 100)
dt = 0.01
tt = tspan[1]:dt:tspan[2]

prob = ODEProblem(lorenz96!, u0, tspan)

function generate_and_save(u0; t0::Int=0, t1::Int=308, dt::Float64=0.01)
    prob = ODEProblem(lorenz96!, u0, (t0, t1))
    sol = solve(prob, Tsit5();saveat=8:dt:t1)
    save_solution(sol, "Tsit")
end

function solve_ensemble(u0, N_ensemble::Int; seed::Int=0, noise_level::Int=1)
    rng = Xoshiro(seed)
    sol = solve(prob, Tsit5(); saveat=tt)
    ensemble = [stack(sol.u; dims=1)]

    for _ in 1:N_ensemble-1
        noise = randn(rng, size(u0))
        prob = ODEProblem(lorenz96!, u0 + noise_level / 100 * noise, tspan)
        push!(ensemble, stack(solve(prob, Tsit5(); saveat=tt).u; dims=1))
    end

    return stack(ensemble; dims=1)
end

function save_solution(sol, name::String)
    mat = stack(sol.u; dims=1)
    npzwrite("Lorenz96/data/$(name).npz", Dict("tt" => sol.t, "sol" => mat))
end

function save_ensemble(ensemble, name::String)
    npzwrite("Lorenz96/data/$(name)_ensemble.npz", Dict("tt" => tt, "ensemble" => ensemble))
end
