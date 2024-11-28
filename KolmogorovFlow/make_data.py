"""
(GPU accelerated) Kolmogorov flow solver.

Modified the original code from https://github.com/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb

Uncommentate `jax.config.update` line if you want to enable double precision.
"""

import time

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import numpy as np
from utils import Configuration, add_noise, initialize_vorticity

from oda.problems import DynamicalCore

print(f"Precision check: {jnp.ones(()).dtype}")
config = Configuration()
print(
    f"Nx: {config.num_grids}, viscosity: {config.viscosity}, noise_level: {config.noise_level}%"
)

# physical parameters
grid = grids.Grid(
    (config.num_grids, config.num_grids), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
)
dt = cfd.equations.stable_time_step(config.max_velocity, 0.5, config.viscosity, grid)
print(f"dt = {dt}")

# setup step function using crank-nicolson runge-kutta order 4
# **use predefined settings for Kolmogorov flow**
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(
        config.viscosity, grid, smooth=config.smooth
    ),
    dt,
)


class KolmogorovFlow(DynamicalCore):
    def __init__(self, dt: float = dt, inner_steps: int = 1):
        super().__init__(
            Nx=config.num_grids, dt=dt, inner_steps=inner_steps
        )  # dt, num_steps
        self._step = jax.jit(step_fn)

    def forecast(self, u0):
        "Repeated application of the `_step` for `self.inner_steps` times."
        u0hat = jnp.fft.rfftn(u0)
        for _ in range(self.inner_steps):
            u0hat = self._step(u0hat)
        return jnp.fft.irfftn(u0hat)


def main(
    observe_every: int = 10,
    burn_steps: int = 1000,
    train_steps: int = 10000,
    test_steps: int = 5000,
    seed: int = 42,
):
    """
    **args**:
    - observe_every: observation occurs every `observe_every`-th step.
    - burn_steps: excluding transient dynamics from training data.
    - train_steps: the length of training trajectory
    - test_steps: the length of test trajectory
    - seed: random seed for generating an initial vorticity.

    **return**:
    - `train.npz` and `test.npz` will be saved in the disk.
    """
    print(
        f"observe every {observe_every}-th step, total {burn_steps + train_steps + test_steps} observations"
    )
    model = KolmogorovFlow(dt=dt, inner_steps=observe_every)
    total_steps = burn_steps + train_steps + test_steps
    tt = np.linspace(0, dt * total_steps * observe_every, total_steps + 1)
    assert np.allclose(
        tt[1] - tt[0], dt * observe_every
    ), "Incorrect number of time steps!"

    vorticity0 = initialize_vorticity(grid, config.max_velocity, seed=seed)
    vorticity0 = add_noise(vorticity0, config.noise_level * 1e-2, seed=0)

    print(f"Solving from t=0 to t={tt[-1]}, Nt = {total_steps * observe_every}, ...")
    tic = time.time()
    uu = model.solve(vorticity0, tt)
    toc = time.time()
    print(f"Done! Elapsed time: {toc - tic:.2f}s")

    uu = np.concatenate([vorticity0[None, ...], uu], 0)

    tt = tt[burn_steps:]
    uu = uu[burn_steps:]
    np.savez("data/train.npz", tt=tt[:-test_steps], sol=uu[:-test_steps])
    assert tt[:-test_steps][-1] == tt[-(test_steps + 1) :][0], "Index Error!"

    tt = tt[-(test_steps + 1) :]
    uu = uu[-(test_steps + 1) :]
    np.savez("data/test.npz", tt=tt, sol=uu)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
