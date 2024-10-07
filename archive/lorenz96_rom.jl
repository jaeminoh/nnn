using Random
using ComponentArrays
using Lux, LuxCUDA
using OrdinaryDiffEq
using DiffEqFlux: NeuralODE
using DALS: lorenz96!
using Optimization
using UnPack
using Plots, LaTeXStrings

# seed
rng = Xoshiro(0)

# gpu setup
CUDA.device!(1) # fourth gpu
const gpu = gpu_device()

# full order model
d = 128
u0 = 8 * ones(d) .+ randn(rng, (d)) * 0.1

u0 = u0 |> gpu # convert to gpu array
tspan = (0.0e0, 5.0e0)
model_full = ODEProblem(lorenz96!, u0, tspan)
sol = solve(model_full, Tsit5())
tsteps = sol.t
data = stack(sol.u, dims=2) |> gpu

# encoder
encoder = Chain(Dense(128, 64, gelu), Dense(64, 32, tanh), Dense(32, 5))
ps, st = Lux.setup(rng, encoder)
ps_e = ps |> ComponentArray
st_e = st |> gpu

# reduced order model
mlp = Chain(Dense(5, 50, tanh), Dense(50, 50, tanh), Dense(50, 5,)) # latent dimension: 5
ps, st = Lux.setup(rng, mlp)
ps_r = ps |> ComponentArray
st_r = st |> gpu
model_reduced = NeuralODE(mlp, tspan, Tsit5(); saveat=tsteps, abstol=1e-3)

# decoder
decoder = Chain(Dense(5, 32, gelu), Dense(32, 64, gelu), Dense(64, 128))
ps, st = Lux.setup(rng, decoder)
ps_d = ps |> ComponentArray
st_d = st |> gpu

# loss
ps = (r=ps_r, e=ps_e, d=ps_d) |> ComponentArray |> gpu .|> Float64
solve_z(z0, p) = stack(first(model_reduced(z0, p, st_r)).u, dims=2)
encode(u, p) = first(encoder(u, p, st_e))
decode(z, p) = first(decoder(z, p, st_d))

function loss(ps)
    @unpack r, e, d = ps
    pred = decode(solve_z(encode(u0, e), r), d)
    l = sum(abs2, pred .- data) / length(pred)
    return l
end

# callback
list_plots = []
iter = 0

function cb(state, l)
    global list_plots, iter
    if iter == 0
        list_plots = []
    end
    iter += 1
    if iter % 10 == 0
        println("iter: $(iter), loss: $(l)")
        # plot current prediction against data
        #plt = scatter(tsteps, Array(data[1, :]); label="data")
        #scatter!(plt, tsteps, Array(pred[1, :]); label="prediction")
        #push!(list_plots, plt)
        #if doplot
        #    display(plot(plt))
        #end
    end

    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((u, p) -> loss(u), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)
t = @elapsed begin
    result = Optimization.solve(optprob, Optimization.Sophia(); callback=cb, maxiters=300)
end
println("Training time: ", t) # 1000s


ps = result.u
uu_pred = decode(solve_z(encode(data[:, 1], ps.e), ps.r), ps.d)

err = uu_pred .- data |> Array
heatmap(err, dpi=300, colormap=:jet, xaxis=L"t", yaxis=L"x")
savefig("figures/lorenz96_rom_latentode.pdf")