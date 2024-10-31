import jax
import jaxopt
import optax
from methods import Euler
from utils import load_data, solve, visualize

from oda.networks import MultiLayerPerceptron, TensorNet

euler = Euler()


def compute_loss(net, u0, yy):
    u_f, u_a = euler.unroll(net, u0, yy)
    loss_3dvar = ((u_f - yy) ** 2).mean()
    return loss_3dvar


def main(
    lr0: float = 1e-3,
    epoch: int = 5000,
    noise_level: int = 0,
):
    # net = MultiLayerPerceptron(d_in=128 * 2, width=128, depth=2, d_out=128)
    net = TensorNet(d_in=128, d_out=128, rank=32)
    opt = optax.adamw(lr0)
    solver = jaxopt.OptaxSolver(compute_loss, opt)
    solver_step = jax.jit(solver.update)

    tt, u0, uu_ref, yy = load_data(60, noise_level=noise_level)
    state = solver.init_state(net, u0, yy)
    net, state, loss_traj = solve(solver_step, net, state, u0, yy, maxiter=epoch)

    # test
    tt, u0, uu_ref, yy = load_data(60, noise_level=noise_level, seed=1)
    uu_base = euler.solve(u0, tt)
    uu_f, uu_a = euler.unroll(net, u0, yy)
    visualize(
        tt,
        uu_ref,
        uu_base,
        uu_f,
        uu_a,
        loss_traj,
        noise_level=noise_level,
        fname="base",
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
