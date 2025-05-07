from dataclasses import dataclass

import jax
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, jaxtyped

from oda.observation import ObservationOperator


@dataclass
class Solution:
    "This holds everything necessary for a visualization."

    tt: ArrayLike
    reference: ArrayLike
    baseline: ArrayLike
    forecast: ArrayLike
    analysis: ArrayLike
    observation: ArrayLike = None

    def save(self, fname: str):
        np.savez(
            f"data/{fname}",
            tt=self.tt,
            uu=self.reference,
            uu_f=self.baseline,
            uu_a=self.forecast,
            yy=self.observation,
        )


class DataLoader:
    """A collection of useful methods for data loading.

    All methods returns `tt`, `u0`, `uu_ref`, and `yy`.

    - `tt`: time points
    - `u0`: initial condition
    - `uu_ref`: reference solution evaluated at `tt[1:]`.
    - `yy`: simulated noisy observation, by adding Gaussian noise to `uu_ref`.
    """

    def __init__(self, observe: ObservationOperator, noise_level: int = 100):
        self.observe = observe
        self.noise_level = noise_level

    @jaxtyped(typechecker=typechecker)
    def load_train(
        self, unroll_length: int = 50, seed: int = 0, max_ens_size=None
    ) -> tuple[
        Float[ArrayLike, " ens *Nx"],
        Float[ArrayLike, " ens {unroll_length} *Nx"],
        Float[ArrayLike, " ens {unroll_length} *No"]
    ]:
        """
        Load a single long time series as an *ensemble* of short time series.

        For instance, let `len(tt) == 30001` and `unroll_length == 50`.
        The long time series will be divided into 600 chunks of length 51 time series,
        where the last element of the current time series (`uu_ref[:-1, -1]`)
        should be equal to the first element of the next time series (`u0[1:]`).
        The chunks are then stacked and regarded as an ensemble.
        """
        np.random.seed(seed)
        d = np.load("data/train.npz")
        # tt = d["tt"]
        uu = d["sol"]
        del d

        Nt, *Nx = uu.shape

        assert (Nt - 1) % unroll_length == 0, "unroll_length should divide (Nt-1)!"
        N_traj = (Nt - 1) // unroll_length

        u0 = np.zeros([N_traj] + Nx)
        uu_ref = np.zeros([N_traj] + [unroll_length] + Nx)

        for i in range(N_traj):
            _start = unroll_length * i
            u0[i] = uu[_start]
            uu_ref[i] = uu[_start + 1 : _start + unroll_length + 1]

        assert np.allclose(u0[1:], uu_ref[:-1, -1]), "index error!"

        u0 = _add_noise(u0, noise_level=self.noise_level)
        yy = _add_noise(jax.vmap(jax.vmap(self.observe))(uu_ref), noise_level=self.noise_level)

        if max_ens_size:
            u0 = u0[:max_ens_size]
            yy = yy[:max_ens_size]
        return u0, uu_ref, yy

    def load_test(self, fname: str, unroll_length: int, seed: int = 1):
        np.random.seed(seed)
        d = np.load(fname)
        tt = d["tt"][: unroll_length + 1]
        uu = d["sol"]
        del d
        u0 = _add_noise(uu[0], noise_level=self.noise_level)
        uu = uu[1 : unroll_length + 1]
        yy = _add_noise(jax.vmap(self.observe)(uu), noise_level=self.noise_level)
        return tt, u0, uu, yy


def _add_noise(target, noise_level: int = 38):
    return target + 0.01 * noise_level * np.random.randn(*target.shape)
