import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from oda.data_containers import Solution


def add_noise(pure_quantity, noise_level: int = 0, seed: int = 0):
    np.random.seed(seed)
    return pure_quantity + 0.01 * noise_level * np.random.randn(*pure_quantity.shape)


def solve(solver_step, net, state, *args, maxiter=5000):
    loss_traj = []
    min_loss = np.inf
    for it in (pbar := trange(1, 1 + maxiter)):
        net, state = solver_step(net, state, *args)
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{(loss:= state.value):.3e}"})
            loss_traj.append(loss)
            if np.isnan(loss):
                break
            elif loss < min_loss:
                min_loss = loss
                opt_net = net

    print(f"Done! min_loss: {min_loss:.3e}, final_loss: {state.value:.3e}")
    return opt_net, state, np.stack(loss_traj)


def load_ensembles(fname: str,
    unroll_length: int = 50, normalization: bool = False, noise_level: int = 0, seed: int = 0
):
    """
    Load reference solution and (simulated) noisy observation, with user-specified unroll-length.

    For example, if unroll_length is 6, then `yy.shape` is `(N_traj, unroll_length, 128)`.
    """
    d = np.load(fname)
    tt = d["tt"]
    uu = d["sol"]
    del d
    Nt, Nx = uu.shape

    assert (Nt - 1) % unroll_length == 0, "unroll_length should divide (Nt-1)!"
    N_traj = (Nt - 1) // unroll_length

    u0 = np.zeros((N_traj, Nx))
    uu_ref = np.zeros((N_traj, unroll_length, Nx))

    for i in range(N_traj):
        _start = unroll_length * i
        u0[i] = uu[_start]
        uu_ref[i] = uu[_start + 1 : _start + unroll_length + 1]

    assert np.allclose(u0[1:], uu_ref[:-1, -1]), "index error!"

    if normalization:
        u0, uu_ref = normalize(u0, uu_ref)

    yy = add_noise(uu_ref, noise_level=noise_level, seed=seed)
    return tt, u0, uu_ref, yy


def load_data(fname:str, N: int, noise_level: int = 0, seed: int = 0):
    """
    Load reference solution and (simulated) noisy observation for `k=0, ..., N`.
    """
    d = np.load(fname)
    tt = d["tt"][: N + 1]
    u0 = d["sol"][0]
    uu = d["sol"][1 : N + 1]
    yy = add_noise(uu, noise_level=noise_level, seed=seed)
    return tt, u0, uu, yy


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
    errors = (
        np.stack([abs(uu_ref - uu_base), abs(uu_ref - uu_f)])
        / scale
    )
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
    plt.savefig(
        f"results/{fname}.pdf",
        format="pdf"
    )


def normalize(*arrays):
    return [(array - 8) / 8 for array in arrays]


def denormalize(*arrays):
    return [(array * 8 + 8) for array in arrays]
