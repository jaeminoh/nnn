using OrdinaryDiffEq, FFTW
using SciMLOperators

N = 256
xx = LinRange(0, 32π, 256 + 1)[begin:end-1]
tspan = (0, 200)
u0(x) = cos(x / 16) * (1 + sin(x / 16))
u0hat = fft(u0.(xx))
k = fftfreq(N, N / (32 * π)) * 2 * π

D = DiagonalOperator((k.^2 - k.^4))
f(uhat, p, t) = 0.5 * (k .* im) .* fft(ifft(uhat).^2)
F = FunctionOperator(f, zeros(N), zeros(N))

prob = SplitODEProblem{false}(D, F, u0hat, tspan)
sol = solve(prob, ETDRK4(), dt=0.25)

using Plots, LaTeXStrings
uu = stack([real(ifft(u)) for u ∈ sol.u], dims=2)
heatmap(uu, xaxis=L"t", yaxis=L"x", colormap=:jet, dpi=300)
savefig("figures/kursiv.pdf")
