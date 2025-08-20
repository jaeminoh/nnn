import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftfreq, rfft, irfft

from nnn.observation import UniformSubsample


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

    @functools.partial(jax.jit, static_argnums=0)
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
        forcing: float = 8.0,
        **kwargs,
    ):
        super().__init__(d_in=d_in, Nx=Nx, dt=dt, inner_steps=inner_steps, **kwargs)
        ii = np.arange(self.Nx)
        self.ii_plus_1 = np.mod(ii + 1, self.Nx)
        self.ii_minus_1 = np.mod(ii - 1, self.Nx)
        self.ii_minus_2 = np.mod(ii - 2, self.Nx)
        self.forcing = forcing

    def __call__(self, u):
        return (
            (u[self.ii_plus_1] - u[self.ii_minus_2]) * u[self.ii_minus_1] - u + self.forcing
        )

    def _step(self, u0):
        return _rk4(self, u0, self.dt)


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
        inner_steps: int = 1,
        method: str = "etdrk4",
        **kwargs,
    ):
        dt = 1 / 4 / inner_steps
        super().__init__(Nx=Nx, dt=dt, inner_steps=inner_steps, **kwargs)
        self.k = 2j * np.pi * jnp.fft.rfftfreq(self.Nx, (xr - xl) / self.Nx)
        self.method = method
        print(f"time stepper: {self.method}")
        if self.method == "etdrk4":
            self._etdrk4_precompute()

    def __call__(self, u):
        linear = jnp.fft.irfft((-(self.k**2) - self.k**4) * jnp.fft.rfft(u), self.Nx)
        nonlinear = -0.5 * jnp.fft.irfft(self.k * jnp.fft.rfft(u**2), self.Nx)
        return linear + nonlinear

    def _step(self, u0):
        if self.method == "etdrk4":
            return irfft(self._etdrk4(rfft(u0)), self.Nx)
        elif self.method == "forward_euler":
            return _forward_euler(self, u0, self.dt)

    def _etdrk4_precompute(self):
        h = self.dt
        k = 2j * np.pi * rfftfreq(self.Nx, 32 * np.pi / self.Nx)
        L = -(k**2) - k**4
        E = np.exp(h * L)
        E2 = np.exp(h * L / 2)
        M = 16  # the number of quadrature points for the contour integral
        r = np.exp(1j * np.pi * np.arange(1 - 0.5, M + 1 - 0.5) / M)
        LR = h * L[:, None] + r
        Q = h * ((np.exp(LR / 2) - 1) / LR).mean(1).real
        f1 = h * ((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(1).real
        f2 = h * ((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3).mean(1).real
        f3 = h * ((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3).mean(1).real

        self.k = k
        self.E = E
        self.E2 = E2
        self.Q = Q
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def _etdrk4(self, v):
        Nv = -0.5 * self.k * rfft(irfft(v, self.Nx) ** 2)
        a = self.E2 * v + self.Q * Nv
        Na = -0.5 * self.k * rfft(irfft(a, self.Nx) ** 2)
        b = self.E2 * v + self.Q * Na
        Nb = -0.5 * self.k * rfft(irfft(b, self.Nx) ** 2)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = -0.5 * self.k * rfft(irfft(c, self.Nx) ** 2)
        v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        return v


def _rk4(f: callable, u0, h: float):
    k1 = f(u0)
    k2 = f(u0 + k1 * h / 2)
    k3 = f(u0 + k2 * h / 2)
    k4 = f(u0 + k3 * h)
    return u0 + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


def _forward_euler(f: callable, u0, h: float):
    return u0 + h * f(u0)
