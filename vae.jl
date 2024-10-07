using Lux, LuxCUDA, Random, Zygote, ComponentArrays, UnPack, Optimisers
using DALS: read_temperature

CUDA.device!(3)
const gpu = gpu_device()
const cpu = cpu_device()
#yy = 1979:1981
#TT = read_temperature.(yy)
temp = read_temperature(2018)[:, :, :]
temp = reshape(temp, 64, 32, 1, size(temp)[3]) |> Array{Float32} |> gpu

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
    return @. exp(logvar / 2) * ϵ + μ
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
    regularization_loss = sum(0.5 * sum(μ .^ 2 .+ exp.(logvar) - logvar .- 1; dims=1)) / length(x)
    return reconstruction_loss + regularization_loss, x̂
end

function optimize(params, x, nepochs::Int=5000; η=1f-2)
    opt_state = Optimisers.setup(Optimisers.ADAM(η), params)
    for i in 1:nepochs
        (loss, x̂), back = pullback(p -> compute_loss(p, x), params)
        grad = back((1, nothing))[1]
        opt_state, params = Optimisers.update(opt_state, params, grad)
        if i % 100 == 0
            println("iter: $(i), loss: $(loss)")
        end
    end
    return opt_state, params
end

opt_state, params = optimize(params, temp; η=1f-3)
μ, logvar = encode(temp[:,:,:,1:1], params.θe)
x̂ = decode(μ, params.θd)
