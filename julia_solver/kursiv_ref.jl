using OrdinaryDiffEq
using FFTW
using Plots, LaTeXStrings
using DALS: kursiv

N = 256
xx = LinRange(0, 32π, N + 1)[begin:end-1]
tspan = (0, 10000)
u0(x) = cos(x / 16) * (1 + sin(x / 16))
u0hat = fft(u0.(xx))
D, F = kursiv(N, 0, 32*π)
prob = SplitODEProblem{false}(D, F, u0hat, tspan)
sol = solve(prob, ETDRK4(), dt=0.25)

uu = stack([real(ifft(u)) for u ∈ sol.u], dims=2)
heatmap(sol.t, xx, uu, xaxis=L"t", yaxis=L"x", colormap=:jet, dpi=300)
savefig("kursiv.pdf")
