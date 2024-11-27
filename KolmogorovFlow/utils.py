from dataclasses import dataclass

import jax.random as jr
import jax_cfd.base as cfd


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

    Return `pure_quantity + scale * z`, z ~ N(0, I)
    """
    return pure_quantity + scale * jr.normal(jr.key(seed), pure_quantity.shape)


@dataclass(frozen=True)
class Configuration:
    viscosity: float = 1e-2
    noise_level: int = 0
    num_grids: int = 64
    max_velocity: float = 7.0
    smooth: bool = True

