import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
from oda.networks import MultiLayerPerceptron
from oda.utils import solve, load_data


def lorenz96(u0):
    N = u0.size
    index = jnp.arange(N)
    n_1 = jnp.mod(index + 1, N)
    n__2 = jnp.mod(index - 2, N)
    n__1 = jnp.mod(index - 1, N)
    return (u0[n_1] - u0[n__2]) * u0[n__1] - u0 + 8


def euler_step(u0, dt=0.01):
    return u0 + dt * lorenz96(u0)


def euler_pred(net, u0, y, dt=0.01):
    u_b = euler_step(u0)
    u_p = u_b + dt * net(u0, y)
    return u_p, jnp.stack([u_b, u_p])


def unroll(net, u0, yy):
    _, out = jax.lax.scan(lambda u0, y: euler_pred(net, u0, y), u0, yy)
    return out[:, 0], out[:, 1]  # u_b, u_p


def compute_loss(net, u0, yy, noise_level: int = 0):
    u_b, u_p = jax.vmap(unroll, (None, 0, 0))(net, u0, yy)
    loss = ((u_p - u_b) ** 2).mean() + ((u_p - yy) ** 2).mean() / (
        0.01 * noise_level + 1e-3
    ) ** 2
    return loss


def main(
    unroll_length: int = 100, lr0: float = 1e-3, epoch: int = 5000, noise_level: int = 0
):
    tt, u0, uu_ref, yy = load_data("Tsit_50", unroll_length, noise_level)
    # observation = original + gaussian noise (10%)
    yy = uu_ref + 0.01 * noise_level * np.random.randn(*uu_ref.shape)

    # training
    net = MultiLayerPerceptron(d_in=128 * 2, width=32, depth=3, d_out=128)
    maxiter = epoch
    opt = optax.adamw(lr0)
    solver = jaxopt.OptaxSolver(
        lambda net, *args: compute_loss(net, u0, yy, noise_level=noise_level), opt
    )
    solver_step = jax.jit(solver.update)
    state = solver.init_state(net)
    net, state = solve(solver_step, net, state, u0, yy, maxiter=maxiter)

    # test
    tt, u0, uu_ref, yy = load_data("Tsit_50", unroll_length // 2 * 3, noise_level)
    uu_base = np.load("data/Euler.npz")["sol"][1: 1 + unroll_length // 2 * 3]
    uu_b, uu_p = unroll(net, u0[0], yy[0])
    fig, (ax, ax0, ax1) = plt.subplots(ncols=3, figsize=(12, 4))
    err = abs(uu_ref[0] - uu_base)
    err0 = abs(uu_ref[0] - uu_b)
    err1 = abs(uu_ref[0] - uu_p)
    scale = abs(uu_ref[0]).max()
    vmax = (np.stack([err, err0, err1])).max()
    ax.imshow(err, vmax=vmax, vmin=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(f"Forward Euler, max: {err.max() / scale:.2e}")
    ax0.imshow(err0, vmax=vmax, vmin=0)
    ax0.axis("off")
    ax0.set_title(f"Prior, max: {err0.max() / scale:.2e}")
    ax1.imshow(err1, vmax=vmax, vmin=0)
    ax1.set_title(f"Posterior, max: {err1.max() / scale:.2e}")
    ax1.axis("off")
    plt.suptitle("Absolute Error")
    plt.tight_layout()
    plt.savefig(
        f"results/unroll{unroll_length}_noise{noise_level}_lr0{lr0:.2e}_epoch{epoch}.pdf",
        dpi=300,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
