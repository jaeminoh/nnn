import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import irfft, rfft, rfftfreq


def lorenz96(u0):
    N = u0.size
    index = jnp.arange(N)
    n_1 = jnp.mod(index + 1, N)
    n__2 = jnp.mod(index - 2, N)
    n__1 = jnp.mod(index - 1, N)
    return (u0[n_1] - u0[n__2]) * u0[n__1] - u0 + 8


class Kursiv:
    def __init__(self, Nx: int = 128, xl: float = 0.0, xr: float = 32 * np.pi):
        self.Nx = Nx
        self.k = 2j * np.pi * rfftfreq(Nx, (xr - xl) / Nx)

    def __call__(self, u):
        linear = irfft((- self.k**2 - self.k**4) * rfft(u), self.Nx)
        nonlinear = -0.5 * irfft(self.k * rfft(u**2), self.Nx)
        return linear + nonlinear
