using OrdinaryDiffEq, Random, Lux, Optimisers, ProgressBars, ParameterSchedulers, Zygote, ComponentArrays
using DALS: lorenz96!, lorenz96

N = 128
u0 = 8 * ones(N)
u0[1] += 0.01

tspan = (0, 200.0)
tt = collect(tspan[begin]:0.01:tspan[end])

prob = ODEProblem(lorenz96!, u0, tspan)
sol_ref = solve(prob, Tsit5(), saveat=tt)

uu_ref = stack(sol_ref.u; dims=2)
rng = Xoshiro(1234)
yy = uu_ref + randn(rng, size(uu_ref)) * 0.1 # observation

MLP = @compact(
    dense = [Dense(2 * N, N, gelu), Dense(N, N, tanh)]) do x
    for layer in dense
        x = layer(x)
    end
    @return x
end

θ, st = Lux.setup(rng, MLP)

θ = θ |> ComponentArray
st = st
yy = yy
u0 = uu_ref[:, begin]

function one_step(θ, u, y; dt::Float32=1.0f-2)
    u_prior = u + dt * lorenz96(u, nothing, nothing)
    u_posterior = u_prior + dt * first(MLP(cat(u, y; dims=1), θ, st))
    return u_prior, u_posterior
end

function predict(θ, u0, yy)
    uu_b = similar(yy)
    uu_p = similar(yy)
    for i in (2:size(yy)[2])
        u0, u_posterior = one_step(θ, u0, yy[:, i])
        uu_b[:, i] .= u0
        uu_p[:, i] .= u_posterior
    end
    return uu_b[:, 2:end], uu_p[:, 2:end]
end

function compute_loss(θ; u0=u0, yy=yy)
    uu_b, uu_p = predict(θ, u0, yy)
    return sum(abs2, uu_p - uu_b) + 100 * sum(abs2, yy[:,2:end] - uu_p)
end

function optimize(θ, nepochs::Int=5000; η::Float32=1.0f-3)
    loss_traj = []
    min_loss = Inf
    opt_state = Optimisers.setup(Optimisers.ADAM(η), θ)
    schedule = ParameterSchedulers.Stateful(CosAnneal(l0=η, l1=0, period=nepochs))
    opt_params = θ
    pbar = tqdm(1:nepochs)
    for i in pbar
        # update model
        loss, back = pullback(compute_loss, θ)
        grad = back(1)[1]
        opt_state, θ = Optimisers.update!(opt_state, θ, grad)
        # update learning rate
        lr = ParameterSchedulers.next!(schedule)
        Optimisers.adjust!(opt_state, lr)

        if i % 100 == 0
            set_postfix(pbar, Loss=loss)
            push!(loss_traj, loss)
            if loss < min_loss
                opt_params = θ
            end
            if isnan(loss)
                break
            end
        end
    end
    return cpu(opt_state), cpu(opt_params), cpu(loss_traj)
end

println("Adam run...")
t = @elapsed begin
    opt_state, θ, loss_traj = optimize(θ, 50000; η=1.0f-3)
end
println("Elapsed time: $(t)s")
