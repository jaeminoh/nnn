import numpy as np
from jax.numpy.fft import irfft, rfft, rfftfreq


class InitialValueProblem:
    "Right-hand-sides of initial value problems written in JAX."

    def __init__(self, Nx: int = 128):
        self.Nx = Nx

    def __call__(self):
        print("Subclass should implement `__call__` method.")


class Lorenz96(InitialValueProblem):
    """
    Lorenz 1996 model.
    """
    def __init__(self, Nx: int = 128):
        super().__init__(Nx=Nx)
        ii = np.arange(self.Nx)
        self.ii_plus_1 = np.mod(ii + 1, self.Nx)
        self.ii_minus_1 = np.mod(ii - 1, self.Nx)
        self.ii_minus_2 = np.mod(ii - 2, self.Nx)

    def __call__(self, u, forcing: float = 8.0):
        return (
            (u[self.ii_plus_1] - u[self.ii_minus_2]) * u[self.ii_minus_1] - u + forcing
        )


class Kursiv(InitialValueProblem):
    """
    Kuramoto-Sivashinsky equation: u_t + uu_x + u_xx + u_xxxx = 0.

    Fourier collocation for spatial discretization.
    """
    def __init__(self, Nx: int = 128, xl: float = 0.0, xr: float = 32 * np.pi):
        super().__init__(Nx=Nx)
        self.k = 2j * np.pi * rfftfreq(self.Nx, (xr - xl) / self.Nx)

    def __call__(self, u):
        linear = irfft((-(self.k**2) - self.k**4) * rfft(u), self.Nx)
        nonlinear = -0.5 * irfft(self.k * rfft(u**2), self.Nx)
        return linear + nonlinear
