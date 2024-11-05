import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def lorenz96(u0):
    N = u0.size
    index = jnp.arange(N)
    n_1 = jnp.mod(index + 1, N)
    n__2 = jnp.mod(index - 2, N)
    n__1 = jnp.mod(index - 1, N)
    return (u0[n_1] - u0[n__2]) * u0[n__1] - u0 + 8


class Lorenz96:
    def __init__(self, *, Nx: int = 128):
        self.u0 = np.hstack([8.01, 8 * np.ones((Nx - 1,))])
        self.Nx = Nx

    def __call__(self, u0):
        index = jnp.arange(self.Nx)
        n_1 = jnp.mod(index + 1, self.Nx)
        n__2 = jnp.mod(index - 2, self.Nx)
        n__1 = jnp.mod(index - 1, self.Nx)
        return (u0[n_1] - u0[n__2]) * u0[n__1] - u0 + 8

    def solve(self, u0, ts: ArrayLike):
        saveat = dfx.SaveAt(ts=ts)
        prob = dfx.ODETerm(lambda t, y, args: self(y))
        solution = dfx.diffeqsolve(
            prob,
            dfx.Tsit5(),
            ts[0],
            ts[-1],
            None,
            u0,
            saveat=saveat,
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
        )
        return solution.ys
