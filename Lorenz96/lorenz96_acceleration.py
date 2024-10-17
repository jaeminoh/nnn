import jax
import jax.numpy as jnp
from networks import MultiLayerPerceptron
import jaxopt
import optax
import time
import numpy as np
import matplotlib.pyplot as plt


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
    u_b = u0 + dt * lorenz96(u0)
    u_p = u_b + dt * net(jnp.hstack([u0, y]))
    return u_p, jnp.stack([u_b, u_p])


def unroll(net, u0, yy):
    _, out = jax.lax.scan(lambda u0, y: euler_pred(net, u0, y), u0, yy)
    return out[:, 0], out[:, 1]  # u_b, u_p


def compute_loss(net, u0, yy):
    u_b, u_p = unroll(net, u0, yy)
    loss = ((u_p - u_b) ** 2).mean() + 100 * ((u_p - yy) ** 2).mean()
    return loss

def main(unroll_length: int, lr0 = 1e-3):
    d = np.load("data/Tsit.npz")
    tt = d["tt"]
    u0, uu_ref = d["sol"][0], d["sol"][1:unroll_length+1]
    uu_base = np.load("data/Euler.npz")["sol"][1:unroll_length+1]
    yy = uu_ref + 0.1 * np.random.randn(
        *uu_ref.shape
    )  # observation = original + gaussian noise (10%)

    net = MultiLayerPerceptron(d_in=128 * 2, width=192, depth=2, d_out=128)

    nIter = 5000
    lr = optax.cosine_decay_schedule(lr0, nIter)
    opt = optax.lion(lr)
    solver = jaxopt.OptaxSolver(compute_loss, opt, maxiter=nIter, verbose=True)

    print("solver running...")
    tic = time.time()
    net, state = solver.run(net, u0, yy)
    toc = time.time()
    print(f"elapsed time: {toc - tic:.2f}, final loss: {state.value:.3e}")

    uu_b, uu_p = unroll(net, u0, yy[:unroll_length])

    fig, (ax, ax0, ax1) = plt.subplots(ncols=3, figsize=(12, 4))
    err = abs(uu_ref - uu_base)
    err0 = abs(uu_ref - uu_b)
    err1 = abs(uu_ref - uu_p)
    vmax = (np.stack([err, err0, err1])).max()
    ax.imshow(err, vmax=vmax, vmin=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(f"Forward Euler, max: {err.max():.2e}")
    ax0.imshow(err0, vmax=vmax, vmin=0)
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$t$")
    ax0.set_title(f"Prior, max: {err0.max():.2e}")
    ax1.imshow(err1, vmax=vmax, vmin=0)
    ax1.set_title(f"Posterior, max: {err1.max():.2e}")
    plt.suptitle("Absolute Error")
    plt.tight_layout()
    plt.savefig(f"outputs/unroll{unroll_length}_lr0{lr0:.2e}.pdf", dpi=300)


if __name__ == "__main__":
    import fire
    fire.Fire(main)