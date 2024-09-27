using FFTW
using Plots, LaTeXStrings
using LatentDA: kursiv

N = 256
xx = LinRange(0, 32π, 256 + 1)[begin:end-1]
tspan = (0, 200)
u0(x) = cos(x / 16) * (1 + sin(x / 16))
u0hat = fft(u0.(xx))
D, F = kursiv(256, 0, 32*π)
prob = SplitODEProblem{false}(D, F, u0hat, tspan)
sol = solve(prob, ETDRK4(), dt=0.25)

uu = stack([real(ifft(u)) for u ∈ sol.u], dims=2)
heatmap(uu, xaxis=L"t", yaxis=L"x", colormap=:jet, dpi=300)
savefig("figures/kursiv.pdf")
