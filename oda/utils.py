import equinox as eqx
import jax
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

from oda.data_utils import DataLoader, Solution
from oda.models import DynamicalCore


def test_on(
    train_or_test: str,
    solver: DynamicalCore,
    net,
    data_loader: DataLoader,
    unroll_length: int = 60
) -> Solution:
    "Test for obtained solution."
    tt, u0, uu_ref, yy = data_loader.load_test(
        f"data/{train_or_test}.npz", unroll_length
    )
    uu_base = solver.solve(u0, tt)
    uu_f, uu_a = solver.unroll(net, u0, yy)
    uu = Solution(tt, uu_ref, uu_base, uu_f, uu_a, yy)
    return uu


def visualize(
    uu: Solution,
    loss_traj: list,
    fname: str = "base",
) -> None:
    "Visualization to assess the quality of solution."
    _, (axs0, axs1) = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    plt.suptitle(fname)

    titles = ["Forward Euler", "Forecast"]
    scale = abs(uu.reference).max()
    errors = (
        np.stack([abs(uu.reference - uu.baseline), abs(uu.reference - uu.forecast)])
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

    indices = [34, 64]
    for i, ax in zip(indices, axs1[:-1]):
        ax.plot(uu.tt[1:], uu.reference[:, i], label="Reference", linewidth=3)
        ax.plot(uu.tt[1:], uu.forecast[:, i], ":", label="Forecast", linewidth=2)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$u(x_i)$")
        ax.set_title(f"{i}th position")

    ax = axs1[-1]
    ax.plot(uu.tt[1:], uu.reference[:, 0], label="Reference", linewidth=3)
    ax.plot(uu.tt[1:], uu.forecast[:, 0], ":", label="Forecast", linewidth=2)
    ax.plot(uu.tt[1:], uu.observation[:, 0], "--", label="Observation", linewidth=1)
    ax.legend()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u(x_i)$")
    ax.set_title(f"{0}th position")

    plt.tight_layout()
    plt.savefig(f"results/{fname}.pdf", format="pdf")


class Optimization:
    def __init__(self, lr0: float = 1e-3, algorithm=optax.lion, epoch: int = int(2e2)):
        lr = optax.cosine_decay_schedule(lr0, epoch)
        self.epoch = epoch
        self.algorithm = algorithm(lr)

    def solve(self, fname: str, model: DynamicalCore, net, data):
        """
        Loops for iterative optimization.

        **args**
        - fname: file name to serialize solution
        - model: Lorenz96? Kursiv? Kolmogorov Flow?
        - net: initial guess of the solution
        - data: inputs for optimization objective.

        **returns**
        - solution
        - loss trajectory
        """
        solver = jaxopt.OptaxSolver(model.compute_loss, self.algorithm)

        u0, yy = data
        state = solver.init_state(net, u0, yy)
        net, state, loss_traj = _solve(
            solver.update, net, state, u0, yy, maxiter=self.epoch
        )

        eqx.tree_serialise_leaves(f"results/{fname}.eqx", net)  # save checkpoint
        return net, loss_traj


def _solve(solver_step, net, state, *args, maxiter: int = 200):
    """
    Iterative minimization.
    
    **args**
        - solver_step: one step of the optimizer
        - net: initial guess for the solution
        - state: state of optimizer (e.g., momentum, past gradients, ...)
        - *args: inputs for optimization objective.
    """
    solver_step = jax.jit(solver_step)
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
