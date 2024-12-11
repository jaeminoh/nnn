import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import xarray as xr
from make_data import KolmogorovFlow

from oda.networks import ConvNet as Net
from oda.utils import DataLoader, Optimization, test_on
from oda.filters import ClassicFilter as Filter


def main(
    lr0: float = 1e-3,
    epoch: int = 200,
    noise_level: int = 75,
    rank: int = 64,
    include_training: bool = True,
    sensor_every: int = 1,
):
    fname = f"kolmogorov_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    assimilate_every = 10
    model = KolmogorovFlow(
        inner_steps=assimilate_every, sensor_every=sensor_every, d_in=2
    )
    filter = Filter(model=model, observe=model.observe)
    net = Net(num_spatial_dim=2, rank=rank, kernel_size=10, stride=sensor_every)
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10, max_ens_size=100)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=5000)
    # uu.save(fname + "_test")

    print(f"""NRMSE.
          w/o assimilation: {np.linalg.norm(uu.baseline - uu.reference) / np.linalg.norm(uu.reference)}
          w/  assimilation: {np.linalg.norm(uu.forecast - uu.reference) / np.linalg.norm(uu.reference)}""")

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
        plt.savefig(f"results/{fname}_{type}.pdf")

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


if __name__ == "__main__":
    import fire

    fire.Fire(main)
