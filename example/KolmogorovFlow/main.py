import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import xarray as xr
from make_data import KolmogorovFlow

from nnn.filters import ClassicFilter as Filter
from nnn.networks import DNO, LinearCorrector
from nnn.utils import DataLoader, Optimization, test_on,  rmse


def main(
        filter_type: str = "nonlinear",
        noise_level: float = 0.75,
        sensor_every: int = 1,
        rank: int = 20,
        lr0: float = 1e-3,
        epoch: int = 100,
        include_training: bool = True,
        unroll_length: int = 10,
        inner_steps: int = 10,
):
    fname = f"KF_{filter_type}_Noise{noise_level}Obs{sensor_every}Rank{rank}"
    print(f"""
=============================
          Configurations:
          filter_type: {filter_type}
          inner_steps: {inner_steps}
          noise_level: {noise_level}
          sensor_every: {sensor_every}
          rank: {rank}
          lr0: {lr0}
          epoch: {epoch}
          include_training: {include_training}
          unroll_length: {unroll_length}
          """)

    model = KolmogorovFlow(
        inner_steps=inner_steps, sensor_every=sensor_every, d_in=2
    )

    if filter_type == "nonlinear":
        net = DNO(
            num_spatial_dims=2,
            Nx=64,
            stride=sensor_every,
            num_channels=rank,
        )

    elif filter_type == "linear":
        net = LinearCorrector(
            d_in=2,
            Nx=64,
            sensor_every=sensor_every,
        )

    filter = Filter(model=model, observe=model.observe)
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.adam, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10, max_ens_size=200)
        net, _ = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)
        # loss_traj = np.ones((epoch // 100,))

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=5000)
    uu.save(fname + "_test")

    print(f"""
          RMSE:  {rmse(uu.forecast, uu.reference)}
          nRMSE: {rmse(uu.forecast, uu.reference, normalize=True)}
=============================""")

    # transform the trajectory into real-space and wrap in xarray for plotting
    tt = uu.tt[1:]
    spatial_coord = np.arange(64) * 2 * np.pi / 64  # same for x and y
    No = 64 // sensor_every
    obs_coord = np.arange(No) * 2 * np.pi / No
    coords = {"time": tt[999::1000], "x": spatial_coord, "y": spatial_coord}
    obs_coords = {"time": tt[999::1000], "x": obs_coord, "y": obs_coord}

    def plotting(type: str, coords=coords):
        if type == "baseline":
            d = uu.baseline
        elif type == "observation":
            d = uu.observation
        elif type == "forecast":
            d = uu.forecast
        elif type == "reference":
            d = uu.reference
        elif type == "base_vs_ref":
            d = uu.baseline - uu.reference
        elif type == "obs_vs_ref":
            d = uu.observation - jax.vmap(model.observe)(uu.reference)
        elif type == "forecast_vs_ref":
            d = uu.forecast - uu.reference
        data = xr.DataArray(d[999::1000], dims=["time", "x", "y"], coords=coords)
        data.plot.imshow(col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True)
        plt.savefig(f"data/{fname}_{type}.pdf")
    
    """
    for t, c in zip(
        [
            "baseline",
            "observation",
            "forecast",
            "reference",
            "base_vs_ref",
            "obs_vs_ref",
            "forecast_vs_ref",
        ],
        [coords, obs_coords, coords, coords, coords, obs_coords, coords],
    ):
        plotting(t, coords=c)
    """

if __name__ == "__main__":
    import fire

    fire.Fire(main)
