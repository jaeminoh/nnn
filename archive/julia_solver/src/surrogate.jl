using Random
using ComponentArrays
using Lux
using OrdinaryDiffEq
using DiffEqFlux: NeuralODE
using UnPack


function init(seed::Int, d_full::Int=128, d_latent::Int=5)
    rng = Xoshiro(seed)

    # networks
    encoder = Chain(Dense(d_full, 64, tanh), Dense(64, 32, tanh), Dense(32, d_latent))
    mlp = Chain(Dense(d_latent, 50, tanh), Dense(50, 50, tanh), Dense(50, d_latent))
    decoder = Chain(Dense(d_latent, 32, tanh), Dense(32, 64, tanh), Dense(64, d_full))

    # initialization
    e, st_e = Lux.setup(rng, encoder)
    n, st_n = Lux.setup(rng, mlp)
    d, st_d = Lux.setup(rng, decoder)
    θ = (e=e, n=n, d=d) |> ComponentArray

    # neural ode in latent space
    tspan = (0, 1)
    neural_ode = NeuralODE(mlp, tspan, Tsit5(); abstol=1e-3)

    # removing st
    ϕ_e(u, p) = first(encoder(u, p, st_e))
    solve_z(z0, p) = first(neural_ode(z0, p, st_n)).u
    ϕ_d(z, p) = first(decoder(z, p, st_d))

    nets = Dict("encoder"=>ϕ_e, "latent_ode"=>solve_z, "decoder"=>ϕ_d)

    return θ, nets
end