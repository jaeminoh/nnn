import jax
import jax.numpy as jnp
import numpy as np

from oda.observation import UniformSubsample


class DynamicalCore:
    "Numerical implementation of a physical model"

    def __init__(
        self,
        d_in: int = 1,
        Nx: int = 128,
        dt: float = 1e-2,
        inner_steps: int = 10,
        sensor_every: int = 1,
    ):
        self.Nx = Nx
        self.dt = dt
        self.inner_steps = inner_steps
        self._step = jax.jit(self._step)  # compile
        self.observe = UniformSubsample(
            num_spatial_dims=d_in, sensor_every=sensor_every
        )

    def __call__(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def forecast(self, u0):
        "Repeated application of the `_step` for `self.inner_steps` times."
        for _ in range(self.inner_steps):
            u0 = self._step(u0)
        return u0

    def solve(self, u0, tt):
        "Solve initial value problem following the time discretization `tt`."
        ulist = [u0]
        for _ in tt[1:] - tt[:-1]:
            ulist.append(self.forecast(ulist[-1]))
        return np.stack(ulist[1:])


class Lorenz96(DynamicalCore):
    """
    Lorenz 1996 model.
    """

    def __init__(
        self,
        d_in: int = 1,
        Nx: int = 128,
        dt: float = 1e-2,
        inner_steps: int = 1,
        **kwargs,
    ):
        super().__init__(d_in=d_in, Nx=Nx, dt=dt, inner_steps=inner_steps, **kwargs)
        ii = np.arange(self.Nx)
        self.ii_plus_1 = np.mod(ii + 1, self.Nx)
        self.ii_minus_1 = np.mod(ii - 1, self.Nx)
        self.ii_minus_2 = np.mod(ii - 2, self.Nx)

    def __call__(self, u, forcing: float = 8.0):
        return (
            (u[self.ii_plus_1] - u[self.ii_minus_2]) * u[self.ii_minus_1] - u + forcing
        )

    def _step(self, u0):
        return u0 + self.dt * self(u0)


class Kursiv(DynamicalCore):
    """
    Kuramoto-Sivashinsky equation: u_t + uu_x + u_xx + u_xxxx = 0.

    Fourier collocation for spatial discretization.
    """

    def __init__(
        self,
        Nx: int = 128,
        xl: float = 0.0,
        xr: float = 32 * np.pi,
        dt: float = 5e-3,
        inner_steps: int = 50,
        **kwargs,
    ):
        super().__init__(Nx=Nx, dt=dt, inner_steps=inner_steps, **kwargs)
        self.k = 2j * np.pi * jnp.fft.rfftfreq(self.Nx, (xr - xl) / self.Nx)

    def __call__(self, u):
        linear = jnp.fft.irfft((-(self.k**2) - self.k**4) * jnp.fft.rfft(u), self.Nx)
        nonlinear = -0.5 * jnp.fft.irfft(self.k * jnp.fft.rfft(u**2), self.Nx)
        return linear + nonlinear

    def _step(self, u0):
        return u0 + self.dt * self(u0)
