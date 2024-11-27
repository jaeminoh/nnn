from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from oda.data_containers import Solution


def solve(solver_step, net, state, *args, maxiter: int = 200):
    loss_traj = []
    min_loss = np.inf
    for it in (pbar := trange(1, 1 + maxiter)):
        net, state = solver_step(net, state, *args)
        if it % 5 == 0:
            pbar.set_postfix({"loss": f"{(loss:= state.value):.3e}"})
            loss_traj.append(loss)
            if np.isnan(loss):
                break
            elif loss < min_loss:
                min_loss = loss
                opt_net = net

    print(f"Done! min_loss: {min_loss:.3e}, final_loss: {state.value:.3e}")
    return opt_net, state, np.stack(loss_traj)


class DataLoader:
    """A collection of useful methods for data loading.

    All methods returns `tt`, `u0`, `uu_ref`, and `yy`.

    - `tt`: time points
    - `u0`: initial condition
    - `uu_ref`: reference solution evaluated at `tt[1:]`.
    - `yy`: simulated noisy observation, by adding Gaussian noise to `uu_ref`.
    """

    def __init__(self, noise_level: int = 100):
        self.noise_level = noise_level

    def load_train(self, unroll_length: int = 50, seed: int = 0):
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
        tt = d["tt"]
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
        yy = _add_noise(uu_ref, noise_level=self.noise_level)
        return tt, u0, uu_ref, yy

    def load_test(self, fname: str, unroll_length: int, seed: int = 1):
        np.random.seed(seed)
        d = np.load(fname)
        tt = d["tt"][: unroll_length + 1]
        uu = d["sol"]
        del d
        u0 = _add_noise(uu[0], noise_level=self.noise_level)
        uu = uu[1 : unroll_length + 1]
        yy = _add_noise(uu, noise_level=self.noise_level)
        return tt, u0, uu, yy


def _add_noise(target, noise_level: int = 0):
    return target + 0.01 * noise_level * np.random.randn(*target.shape)


def visualize(
    uu: Solution,
    loss_traj,
    fname: str = "base",
):
    uu_ref = uu.reference
    uu_base = uu.baseline
    uu_f = uu.forecast
    uu_a = uu.analysis
    tt = uu.tt
    yy = uu.observation

    _, (axs0, axs1) = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    plt.suptitle(fname)

    titles = ["Forward Euler", "Forecast"]
    scale = abs(uu_ref).max()
    errors = np.stack([abs(uu_ref - uu_base), abs(uu_ref - uu_f)]) / scale
    vmax = errors[1].max()
    for ax, title, error in zip(axs0[:-1], titles, errors):
        ax.imshow(error, vmax=vmax, vmin=0, aspect="auto")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_title(f"{title}, max: {error.max():.2e}")

    ax = axs0[-1]
    ax.semilogy(loss_traj)
    ax.set_title(f"Learning Curve, min: {loss_traj.min():.3e}")
    ax.set_xlabel("100 Iterations")

    indices = [0, 64]
    for i, ax in zip(indices, axs1[:-1]):
        ax.plot(tt[1:], uu_ref[:, i], label="Reference", linewidth=3)
        ax.plot(tt[1:], uu_f[:, i], ":", label="Forecast", linewidth=2)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$u(x_i)$")
        ax.set_title(f"{i}th position")

    ax = axs1[-1]
    ax.plot(tt[1:], uu_ref[:, 32], label="Reference", linewidth=3)
    ax.plot(tt[1:], uu_f[:, 32], ":", label="Forecast", linewidth=2)
    ax.plot(tt[1:], yy[:, 32], "--", label="Observation", linewidth=1)
    ax.legend()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u(x_i)$")
    ax.set_title(f"{32}nd position")

    plt.tight_layout()
    plt.savefig(f"results/{fname}.pdf", format="pdf")


class InitialValueProblemUtil:
    def __init__(self, dt: float = 0.01, inner_steps: int = 1):
        self.dt = dt
        self.inner_steps = inner_steps

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
        return u_f + self.dt * self.inner_steps * net(u_f, y)

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


class Euler(InitialValueProblemUtil):
    def __init__(self, problem, dt: float = 0.01, inner_steps: int = 1):
        super().__init__(dt=dt, inner_steps=inner_steps)
        self.problem = problem

    @partial(jax.jit, static_argnums=0)
    def _step(self, u0):
        return u0 + self.dt * self.problem(u0)
