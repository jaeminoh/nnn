import jax
import jaxopt
import equinox as eqx
import optax
from methods import Euler
from utils import load_data, load_ensembles, solve, visualize

from oda.data_containers import Solution
from oda.networks import TensorNet

euler = Euler()


def compute_loss(net, u0, yy):
    u_f, u_a = jax.vmap(euler.unroll, (None, 0, 0))(net, u0, yy)
    loss_3dvar = ((u_f - yy) ** 2).mean()
    return loss_3dvar


def main(lr0: float = 1e-3, epoch: int = 100000, noise_level: int = 0, rank: int = 32):
    fname = f"ensembles_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"

    net = TensorNet(d_in=128, d_out=128, rank=rank)
    lr = optax.cosine_decay_schedule(lr0, epoch)
    opt = optax.adamw(lr)
    solver = jaxopt.OptaxSolver(compute_loss, opt)
    solver_step = jax.jit(solver.update)

    # train
    _, u0, _, yy = load_ensembles(60, normalization=False, noise_level=noise_level)
    state = solver.init_state(net, u0, yy)
    net, state, loss_traj = solve(solver_step, net, state, u0, yy, maxiter=epoch)
    
    # save checkpoint
    eqx.tree_serialise_leaves(f"results/{fname}.eqx", net)

    # test
    tt, u0, uu_ref, yy = load_data(120, noise_level=noise_level, seed=1)
    uu_base = euler.solve(u0, tt)
    uu_f, uu_a = euler.unroll(net, u0, yy)
    uu = Solution(tt, uu_ref, uu_base, uu_f, uu_a)
    visualize(uu, loss_traj, noise_level=noise_level, fname=fname)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
