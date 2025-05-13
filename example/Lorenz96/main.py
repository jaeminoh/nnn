import equinox as eqx
import numpy as np
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from oda.filters import ClassicFilter as Filter
from oda.models import Lorenz96
from oda.networks import DNO as Net
from oda.utils import DataLoader, Optimization, test_on, visualize, rmse


def main(
        Nx: int = 40,
        forcing: float = 8.0,
        noise_level: float = 0.364,
        sensor_every: int = 1,
        rank: int = 20,
        lr0: float = 1e-3,
        epoch: int = 200,
        include_training: bool = True,
        test_unroll_length: int = 100,
        unroll_length: int = 3,
):
    fname = f"L96_Forcing{int(forcing)}Noise{noise_level}Obs{sensor_every}Rank{rank}Nx{Nx}"
    print(f"""Configurations:
          forcing: {forcing}
          noise_level: {noise_level}
          sensor_every: {sensor_every}
          rank: {rank}
          Nx: {Nx}
          lr0: {lr0}
          epoch: {epoch}
          include_training: {include_training}
          unroll_length: {unroll_length}
          """)
    assimilation_window = 15
    model = Lorenz96(d_in=1, Nx=Nx, sensor_every=sensor_every, inner_steps=assimilation_window, forcing=forcing)
    filter = Filter(model=model, observe=model.observe)
    #net = Net(
    #    hidden_channels=rank, kernel_size=5, stride=sensor_every, num_spatial_dim=1
    #)
    net = Net(stride=sensor_every, Nx=Nx, num_channels=rank)
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.adam, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=unroll_length)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on(
        "train", filter, net, data_loader=data_loader, unroll_length=test_unroll_length
    )
    visualize(uu, loss_traj, fname=fname + "_train")

    uu = test_on(
        "test", filter, net, data_loader=data_loader, unroll_length=test_unroll_length
    )
    uu.save(fname + "_test")
    visualize(uu, loss_traj, fname=fname + "_test")

    plt.cla()
    plt.figure()
    norm_reference = jax.vmap(jnp.linalg.norm)(uu.reference)
    err_baseline = jax.vmap(jnp.linalg.norm)(uu.baseline - uu.reference) / norm_reference
    err_forecast = jax.vmap(jnp.linalg.norm)(uu.forecast - uu.reference) / norm_reference
    plt.semilogy(uu.tt[1:], err_baseline, label="without assimilation")
    plt.semilogy(uu.tt[1:], err_forecast, "--", label="with assimilation")
    plt.xlabel("time")
    plt.ylabel(r"Relative $L^2$ error")
    plt.tight_layout()
    plt.legend()
    plt.savefig("data/" + fname + "_err.pdf", dpi=300)

    print(f"""
          (RMSE, nRMSE)
          {rmse(uu.forecast, uu.reference)}, {rmse(uu.forecast, uu.reference, normalize=True)}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
