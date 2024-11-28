import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import xarray as xr
from make_data import KolmogorovFlow

from oda.networks import ConvNet2d
from oda.utils import DataLoader, test_on, Optimization



def main(
    lr0: float = 1e-3,
    epoch: int = 200,
    noise_level: int = 75,
    rank: int = 64,
    include_training: bool = True,
):
    fname = f"kolmogorov_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    assimilate_every = 10
    model = KolmogorovFlow(inner_steps=assimilate_every)
    net = ConvNet2d(rank=rank, kernel_size=10)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        data_loader = DataLoader(noise_level=noise_level)
        _, u0, _, yy = data_loader.load_train(unroll_length=10)
        net, loss_traj = opt.train(fname, model, net, [u0, yy])
        del u0, yy
    else:
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("test", model, noise_level, net, unroll_length=5000)
    # uu.save(fname + "_test")

    print(f"""NRMSE.
          w/o assimilation: {np.linalg.norm(uu.baseline - uu.reference) / np.linalg.norm(uu.reference)}
          w/  assimilation: {np.linalg.norm(uu.forecast - uu.reference) / np.linalg.norm(uu.reference)}""")

    # transform the trajectory into real-space and wrap in xarray for plotting
    tt = uu.tt[1:]
    spatial_coord = np.arange(64) * 2 * np.pi / 64  # same for x and y
    coords = {
        "time": tt[999::1000],
        "x": spatial_coord,
        "y": spatial_coord,
    }

    def plotting(type: str):
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
            d = uu.observation - uu.reference
        elif type == "forecast_vs_ref":
            d = uu.forecast - uu.reference
        data = xr.DataArray(d[999::1000], dims=["time", "x", "y"], coords=coords)
        data.plot.imshow(col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True)
        plt.savefig(f"results/{fname}_{type}.pdf")

    for t in [
        "baseline",
        "observation",
        "forecast",
        "reference",
        "base_vs_ref",
        "obs_vs_ref",
        "forecast_vs_ref",
    ]:
        plotting(t)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
