import os

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike, Float, jaxtyped
from beartype import beartype as typechecker
import optax
import jaxopt
import matplotlib.pyplot as plt

from oda.models import Lorenz96, _rk4
from oda.networks import DNO as Net
from oda.utils import DataLoader, _solve, rmse


def main(
    lr0: float = 1e-3,
    epoch: int = 50,
    noise_level: int = 1,
    rank: int = 40,
    include_training: bool = True,
    sensor_every: int = 2,
    test_unroll_length: int = 100,
    Nx: int = 40,
):
    fname = f"lorenz_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    model = Lorenz96(d_in=1, Nx=Nx, sensor_every=sensor_every, inner_steps=10, dt=0.1)

    net = Net(num_channels=rank, stride=sensor_every)
    try:
        data_loader = DataLoader(model.observe, noise_level=noise_level)
    except FileNotFoundError:
        os.system("python make_data.py")
        data_loader = DataLoader(model.observe, noise_level=noise_level)



    @jaxtyped(typechecker=typechecker)
    def _step(
        net: eqx.Module,
        u0: Float[ArrayLike, " Nx"],
        y: Float[ArrayLike, " No"],
        dt: float = model.dt,
    ) -> Float[ArrayLike, " Nx"]:
        #Hu = model.observe(u0)
        return _rk4(lambda u: model(u) + net(u, y - model.observe(u)), u0, dt)  # uhat

    def _forecast(net, u0, y, dt):
        u = _step(net, u0, y, dt)
        return u, u

    def forecast(net, u0, y, dt=model.dt, inner_steps: int = 10):
        u, _ = jax.lax.scan(
            lambda carry, _: _forecast(net, carry, y, dt / inner_steps),
            u0,
            None,
            length=inner_steps,
        )
        return u, u

    def unroll(net, u0, yy, dt=model.dt, assimilation_window: int = 10):
        _, uu = jax.lax.scan(
            lambda u, y: forecast(net, u, y, dt=dt, inner_steps=assimilation_window),
            u0,
            yy,
        )
        return uu

    def _loss(net, u0, yy, dt=model.dt, assimilation_window: int = 10):
        uu = unroll(net, u0, yy, dt, assimilation_window)
        return jnp.mean((jax.vmap(model.observe)(uu) - yy) ** 2)

    @jaxtyped(typechecker=typechecker)
    def loss(
        net,
        u0: Float[ArrayLike, "Ne Nx"],
        yy: Float[ArrayLike, "Ne Nt No"],
        dt=model.dt,
        assimilation_window: int = 10,
    ):
        return jax.vmap(_loss, in_axes=(None, 0, 0, None, None))(
            net, u0, yy, dt, assimilation_window
        ).mean()

    if include_training:
        solver = jaxopt.OptaxSolver(loss, optax.lion(lr0))
        u0, yy = data_loader.load_train(unroll_length=10)
        state = solver.init_state(net, u0, yy)
        net, state, _ = _solve(solver.update, net, state, u0, yy, maxiter=epoch)
        eqx.tree_serialise_leaves(f"data/{fname}.eqx", net)
    else:
        net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)

    test_data = data_loader.load_test("data/test.npz", unroll_length=test_unroll_length)
    tt, u0, uu, yy = test_data
    uu_analysis = unroll(net, u0, yy, dt=model.dt, assimilation_window=10)
    plt.plot(tt[1:], uu_analysis[:, 1], label="analysis")
    plt.plot(tt[1:], uu[:, 1], label="reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + "_analysis.pdf")
    print(f"""aRMSE: {rmse(uu_analysis, uu)}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
