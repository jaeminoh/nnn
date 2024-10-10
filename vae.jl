using Lux, LuxCUDA, Random, Zygote, ComponentArrays, UnPack, Optimisers, ProgressBars, ParameterSchedulers
using DALS: read_normalized_daily_temperature

CUDA.device!(1)
const gpu = gpu_device()
const cpu = cpu_device()
println("setup gpu: $(gpu)")

tt, dd, μ, σ = read_normalized_daily_temperature(1979, 2018)
tt = reshape(tt, size(tt)[1:2]..., 1, prod(size(dd))) |> Array{Float32} |> gpu
println("load data: $(size(tt))")

d_latent = 100

encoder = @compact(
    conv = [
        Conv((3, 3), 1 => 5, elu, stride=(2, 2), pad=SamePad()),
        Conv((3, 3), 5 => 10, elu, stride=(2, 2), pad=SamePad()),
        Conv((3, 3), 10 => 20, elu, stride=(2, 2), pad=SamePad()),
        Conv((3, 3), 20 => 40, elu, stride=(2, 2), pad=SamePad()),
        Conv((3, 3), 40 => 80, stride=(1, 2), pad=SamePad())
    ],
    flat = FlattenLayer(),
    dense = Dense(320, 2 * d_latent)) do x
    for cnn in conv
        x = cnn(x)
    end
    x = dense(flat(x))
    @return x[begin:100, :], x[101:end, :] # mean and logvar
end

decoder = @compact(
    dense = Dense(d_latent, 2 * 2 * 80, relu), # (45, 45, 80) -> (720, 1440)
    conv = [
        ConvTranspose((3, 3), 80 => 40, elu, stride=(2, 1), pad=SamePad()),
        ConvTranspose((3, 3), 40 => 20, elu, stride=(2, 2), pad=SamePad()),
        ConvTranspose((3, 3), 20 => 10, elu, stride=(2, 2), pad=SamePad()),
        ConvTranspose((3, 3), 10 => 5, elu, stride=(2, 2), pad=SamePad()),
        ConvTranspose((3, 3), 5 => 1, stride=(2, 2), pad=SamePad())
    ]) do x
    embed = dense(x)
    embed = reshape(embed, (2, 2, 80, size(embed)[end]))
    for cnn in conv
        embed = cnn(embed)
    end
    @return embed
end

function reparametrize(μ, logvar)
    ϵ = randn!(similar(μ))
    return ϵ .* exp.(logvar / 2) + μ
end

rng = Xoshiro(0)
θe, st_e = Lux.setup(rng, encoder)
θd, st_d = Lux.setup(rng, decoder)
params = (θe=θe, θd=θd) |> ComponentArray |> gpu
st = (st_e=st_e, st_d=st_d) |> ComponentArray |> gpu

encode(x, θ) = first(encoder(x, θ, st_e))
decode(z, θ) = first(decoder(z, θ, st_d))

function compute_loss(params, x)
    @unpack θe, θd = params
    μ, logvar = encode(x, θe)
    z = reparametrize(μ, logvar)
    x̂ = decode(z, θd)
    reconstruction_loss = sum(abs2, x̂ - x) / length(x)
    regularization_loss = sum(sum(μ .^ 2 .+ exp.(logvar) - logvar .- 1; dims=1) / 2) / length(x)
    return 1f3*reconstruction_loss + regularization_loss, x̂
end

function optimize(params, x, nepochs::Int=5000; η=1.0f-3)
    loss_traj = []
    min_loss = Inf
    opt_state = Optimisers.setup(Optimisers.ADAM(η), params)
    schedule = ParameterSchedulers.Stateful(CosAnneal(l0=η, l1=0, period=nepochs))
    opt_params = params
    pbar = tqdm(1:nepochs)
    for i in pbar
        # update model
        (loss, x̂), back = pullback(p -> compute_loss(p, x), params)
        grad = back((1, nothing))[1]
        opt_state, params = Optimisers.update!(opt_state, params, grad)
        # update learning rate
        lr = ParameterSchedulers.next!(schedule)
        Optimisers.adjust!(opt_state, lr)

        if i % 100 == 0
            set_postfix(pbar, Loss=loss)
            push!(loss_traj, loss)
            if loss < min_loss
                opt_params = params
            end
            if isnan(loss)
                break
            end
        end
    end
    return opt_state, opt_params, loss_traj
end

println("Adam run...")
t = @elapsed begin
    opt_state, params, loss_traj = optimize(params, tt, 50000; η=1.0f-3)
end
println("Elapsed time: $(t)s")

μ, logvar = encode(tt[:, :, :, 1:1], params.θe)
x̂ = decode(μ, params.θd)

using Plots
plot(loss_traj, yaxis=:log10, legend=false, xlabel="100 iters")
savefig("figures/learning_curve.pdf")

heatmap(cpu(x̂)[:, :, 1, 1])
savefig("figures/x_recon.pdf")

heatmap(cpu(tt[:, :, 1, 1]))
savefig("figures/x_era5.pdf")

using JLD2
params = cpu(params)
st = cpu(st)
@save "checkpoint/test.jld2" params st
@unpack params, st = load("checkpoint/test.jld2")
