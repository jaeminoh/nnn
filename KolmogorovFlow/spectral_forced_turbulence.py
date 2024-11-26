"""
(GPU accelerated) Kolmogorov flow solver.

Modified the original code from https://github.com/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb
"""

import time

import jax.numpy as jnp
import jax.random as jr
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import matplotlib.pyplot as plt
import seaborn as sns
import xarray


def initialize_vorticity(grid, max_velocity, seed: int = 42):
    """
    create an initial velocity field and compute the fft of the vorticity.
    the spectral code assumes an fft'd vorticity for an initial state
    """
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jr.key(seed), grid, max_velocity, 4
    )
    return cfd.finite_differences.curl_2d(v0).data


def add_noise(pure_quantity, scale: float, seed: int = 0):
    """
    Corrupt `pure_quantity` with mean zero Gaussian noise.

    args:
    - pure_quantity: to be corrupted
    - scale: `pure_quantity + scale * z`
    """
    return pure_quantity + scale * jr.normal(jr.key(seed), pure_quantity.shape)


def main(num_grids: int = 256, noise_scale: int = 0):
    print(f"Kolmogorov Flow. Nx = {num_grids}, noise_scale = {noise_scale}")
    fname = f"forced_turbulence_{num_grids}x{num_grids}"
    # physical parameters
    viscosity = 1e-3
    max_velocity = 7
    grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    dt = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid)

    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing

    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt
    )

    # run the simulation up until time 25.0 but only save 10 frames for visualization
    final_time = 25.0
    outer_steps = 10
    inner_steps = (final_time // dt) // 10

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
    )

    vorticity0 = initialize_vorticity(grid, max_velocity, seed=42)
    vorticity0 = add_noise(vorticity0, noise_scale * 1e-2, seed=0)
    fname += f"_noise{noise_scale}"
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    tic = time.time()
    _, trajectory = trajectory_fn(vorticity_hat0)
    toc = time.time()
    print(f"Solved! Elapsed time: {toc - tic:.2f}s")

    # transform the trajectory into real-space and wrap in xarray for plotting
    spatial_coord = (
        jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
    )  # same for x and y
    coords = {
        "time": dt * jnp.arange(outer_steps) * inner_steps,
        "x": spatial_coord,
        "y": spatial_coord,
    }
    xarray.DataArray(
        jnp.fft.irfftn(trajectory, axes=(1, 2)), dims=["time", "x", "y"], coords=coords
    ).plot.imshow(col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True)

    plt.savefig(f"{fname}.pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
