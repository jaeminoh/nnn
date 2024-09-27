using Random
using ComponentArrays
using Lux, LuxCUDA
using OrdinaryDiffEq
using DiffEqFlux: NeuralODE
using LatentDA: lorenz96!
using Optimization, OptimizationOptimisers
using Plots

# gpu setup
CUDA.device!(3)
const gpu = gpu_device()


# full order model
u0 = 8 * ones(128) |> Array{Float32}
u0[1] += 0.01
u0 = u0 |> gpu # convert to gpu array
tspan = (0.0f0, 1.0f1)
model_full = ODEProblem(lorenz96!, u0, tspan)
sol = solve(model_full, Tsit5())
data = stack(sol.u, dims=2) |> gpu

# reduced order model
rng = Xoshiro(0)
mlp = Chain(Dense(5, 50, tanh), Dense(50, 5)) # latent dimension: 5
ps, st = Lux.setup(rng, mlp)
ps_r = ps |> ComponentArray
st_r = st |> gpu
model_reduced = NeuralODE(mlp, tspan, Tsit5(); saveat=sol.t)

# encoder
encoder = Chain(Dense(128, 50, tanh), Dense(50, 5))
ps, st = Lux.setup(rng, encoder)
ps_e = ps |> ComponentArray
st_e = st |> gpu

# decoder
decoder = Chain(Dense(5, 50, tanh), Dense(50, 128))
ps, st = Lux.setup(rng, decoder)
ps_d = ps |> ComponentArray
st_d = st |> gpu

# loss
ps = (r=ps_r, e=ps_e, d=ps_d) |> ComponentArray |> gpu
predict_model(z0, p) = stack(first(model_reduced(z0, p, st_r)).u, dims=2)
encode_u(u, p) = first(encoder(u, p, st_e))
decode_z(z, p) = first(decoder(z, p, st_d))

function loss(ps)
    code = encode_u(data, ps.e)
    zz = predict_model(code[:, 1], ps.r)
    encoder_loss = sum(abs2, zz .- code)
    uu_pred = decode_z(zz, ps.d)
    decoder_loss = sum(abs2, uu_pred .- data)
    return encoder_loss + decoder_loss
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)
result = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); maxiters=50)

ps = result.u
uu_pred = decode_z(predict_model(encode_u(data[:, 1], ps.e), ps.r), ps.d)

err = uu_pred .- data |> Array
heatmap(err)
