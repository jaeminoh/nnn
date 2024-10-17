using FFTW
using SciMLOperators

function lorenz96!(du, u, p, t) # gpu-compatiable implementation
    N = length(du)
    n = 1:N
    @. du = (u[mod1(n + 1, N)] - u[mod1(n - 2, N)]) * u[mod1(n - 1, N)] - u[n] + 8
end

function lorenz96(u, p, t)
    N = length(u)
    n = 1:N
    du = @. (u[mod1(n + 1, N)] - u[mod1(n - 2, N)]) * u[mod1(n - 1, N)] - u[n] + 8
    return du
end

function kursiv(N=256, xl=0, xr=32 * π)
    k = fftfreq(N, N / (xr - xl)) * 2 * π
    D = DiagonalOperator((k .^ 2 - k .^ 4))
    f(uhat, p, t) = 0.5 * (k .* im) .* fft(ifft(uhat) .^ 2)
    F = FunctionOperator(f, zeros(N), zeros(N))
    return D, F
end
