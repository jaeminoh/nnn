import equinox as eqx
import jax
import jaxopt
import numpy as np
import optax
from methods import Euler
from utils import load_data, load_ensembles, solve, visualize

from oda.data_containers import Solution
from oda.networks import ConvNet

euler = Euler()


def compute_loss(net, u0, yy):
    u_f, u_a = jax.vmap(euler.unroll, (None, 0, 0))(net, u0, yy)
    loss_4dvar = ((u_a[:, 0] - yy[:, 0]) ** 2).mean() + (
        (u_f[:, 1:] - yy[:, 1:]) ** 2
    ).mean()
    return loss_4dvar


def train(
    fname: str,
    lr0: float = 1e-3,
    epoch: int = 100000,
    noise_level: int = 0,
    rank: int = 32,
):
    net = ConvNet(d_in=128, rank=rank, kernel_size=10)
    lr = optax.cosine_decay_schedule(lr0, epoch)
    opt = optax.lion(lr)
    solver = jaxopt.OptaxSolver(compute_loss, opt)
    solver_step = jax.jit(solver.update)

    # train
    _, u0, _, yy = load_ensembles(10, normalization=False, noise_level=noise_level)
    state = solver.init_state(net, u0, yy)
    net, state, loss_traj = solve(solver_step, net, state, u0, yy, maxiter=epoch)

    # save checkpoint
    eqx.tree_serialise_leaves(f"results/{fname}.eqx", net)
    return net, loss_traj


def test_on(set: str, noise_level, net, unroll_length: int = 60):
    if set == "test":
        tt, u0, uu_ref, yy = load_data(
            unroll_length, noise_level=noise_level, seed=1, is_train=False
        )
    else:
        tt, u0, uu_ref, yy = load_data(
            unroll_length, noise_level=noise_level, seed=1, is_train=True
        )
    uu_base = euler.solve(u0, tt)
    uu_f, uu_a = euler.unroll(net, u0, yy)
    uu = Solution(tt, uu_ref, uu_base, uu_f, uu_a)
    return uu


def main(
    lr0: float = 1e-3,
    epoch: int = 100000,
    noise_level: int = 0,
    rank: int = 32,
    include_training: bool = True,
):
    fname = f"ensembles_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    if include_training:
        net, loss_traj = train(
            fname, lr0=lr0, epoch=epoch, noise_level=noise_level, rank=rank
        )
    else:
        net = ConvNet(d_in=128, rank=rank, kernel_size=10)
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("train", noise_level, net, unroll_length=30)
    visualize(uu, loss_traj, fname=fname+"_train")

    uu = test_on("test", noise_level, net, unroll_length=30)
    visualize(uu, loss_traj, fname=fname+"_test")

if __name__ == "__main__":
    import fire

    fire.Fire(main)
