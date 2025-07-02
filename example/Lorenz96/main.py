import equinox as eqx
import numpy as np
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from oda.filters import ClassicFilter as Filter
from oda.models import Lorenz96
from oda.networks import DNO, LinearCorrector
from oda.utils import DataLoader, Optimization, test_on, visualize, rmse


def main(
        Nx: int,
        forcing: float,
        noise_level: float,
        filter_type: str = "nonlinear",
        sensor_every: int = 1,
        rank: int = 20,
        lr0: float = 1e-3,
        epoch: int = 300,
        include_training: bool = True,
        test_unroll_length: int = 400,
        unroll_length: int = 5,
):
    fname = f"L96_{filter_type}_Forcing{int(forcing)}Noise{noise_level}Obs{sensor_every}Nx{Nx}"
    print(f"""
          ==============================
          Configurations:
          filter_type: {filter_type}
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
    if filter_type == "nonlinear":
        net = DNO(
            stride=sensor_every,
            Nx=Nx,
            num_channels=rank,
            num_spatial_dims=1,
        )
    elif filter_type == "linear":
        net = LinearCorrector(
            d_in=1,
            Nx=Nx,
            sensor_every=sensor_every,
        )
    data_loader = DataLoader(model.observe, noise_level=noise_level)  

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.adam, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=unroll_length)
        #u0, uu, _ = train_data
        #uu = np.concatenate([u0[:,None,:], uu], axis=1)
        #filter.mean = np.mean(uu, axis=(0,1))
        #filter.std = np.std(uu, axis=(0,1)) + 1e-3
        #del u0, uu
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
          RMSE:  {rmse(uu.forecast, uu.reference)}
          nRMSE: {rmse(uu.forecast, uu.reference, normalize=True)}
          =============================""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
