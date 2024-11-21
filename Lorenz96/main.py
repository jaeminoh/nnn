import equinox as eqx
import jax
import jaxopt
import numpy as np
import optax

from oda.data_containers import Solution
from oda.methods import Euler
from oda.networks import ConvNet
from oda.problems import Lorenz96
from oda.utils import DataLoader, solve, visualize

euler = Euler(Lorenz96(), dt=0.01)


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
    net = ConvNet(d_in=128, rank=rank)
    lr = optax.cosine_decay_schedule(lr0, epoch)
    opt = optax.adamw(lr)
    solver = jaxopt.OptaxSolver(compute_loss, opt)
    solver_step = jax.jit(solver.update)

    # train
    data_loader = DataLoader(noise_level)
    _, u0, _, yy = data_loader.load_train(unroll_length=60)
    state = solver.init_state(net, u0, yy)
    net, state, loss_traj = solve(solver_step, net, state, u0, yy, maxiter=epoch)

    # save checkpoint
    eqx.tree_serialise_leaves(f"results/{fname}.eqx", net)
    return net, loss_traj


def test_on(set: str, noise_level, net, unroll_length: int = 60):
    data_loader = DataLoader(noise_level)
    tt, u0, uu_ref, yy = data_loader.load_test(f"data/{set}.npz", unroll_length)
    uu_base = euler.solve(u0, tt)
    uu_f, uu_a = euler.unroll(net, u0, yy)
    uu = Solution(tt, uu_ref, uu_base, uu_f, uu_a, yy)
    return uu


def main(
    lr0: float = 1e-3,
    epoch: int = 100000,
    noise_level: int = 0,
    rank: int = 32,
    include_training: bool = True,
):
    fname = f"lorenz_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    if include_training:
        net, loss_traj = train(
            fname, lr0=lr0, epoch=epoch, noise_level=noise_level, rank=rank
        )
    else:
        net = ConvNet(d_in=128, rank=rank)
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("train", noise_level, net, unroll_length=1000)
    visualize(uu, loss_traj, fname=fname + "_train")

    uu = test_on("test", noise_level, net, unroll_length=1000)
    uu.save(fname + "_test")
    visualize(uu, loss_traj, fname=fname + "_test")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
