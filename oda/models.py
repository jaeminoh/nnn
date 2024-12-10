import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, jaxtyped

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

    def analysis(self, net, u_f, y):
        """
        Neural filtering (or correction) of forecast `u_f` based on observation `y`.
        Returns analysis `u_a`.
        """
        return u_f + self.dt * self.inner_steps * net(u_f, self.observe(u_f), y)

    def _scan_fn(self, net, u0, y):
        u_f = self.forecast(u0)
        u_a = self.analysis(net, u_f, y)
        return u_a, jnp.stack([u_f, u_a])

    def unroll(self, net, u0, yy):
        """
        Fast (differentiable) for-loop for forecast and analysis.
        Returns `u_f` and `u_a`.

        The number of iterations, the number of rows of `out` and `yy` are the same.
        """
        _, out = jax.lax.scan(lambda u0, y: self._scan_fn(net, u0, y), u0, yy)
        return out[:, 0], out[:, 1]  # u_f, u_a

    def _compute_loss(
        self, net, u0: Float[ArrayLike, " *Nx"], yy: Float[ArrayLike, " Nt ..."]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = self.observe(u_a[0]) - yy[0]
        j1 = jax.vmap(self.observe)(u_f[1:]) - yy[1:]
        return j0, j1

    @jaxtyped(typechecker=typechecker)
    def compute_loss(
        self, net, u0: Float[ArrayLike, " Ne *Nx"], yy: Float[ArrayLike, " Ne Nt ..."]
    ) -> Float[ArrayLike, ""]:
        j0, j1 = jax.vmap(lambda u0, yy: self._compute_loss(net, u0, yy), (0, 0))(
            u0, yy
        )
        return (j0**2).mean() + (j1**2).mean()


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
