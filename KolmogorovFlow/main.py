import equinox as eqx
import jax
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import xarray as xr
from make_data import CrankNicolsonRK4

from oda.data_containers import Solution
from oda.networks import ConvNet2d
from oda.utils import DataLoader, solve

assimilate_every = 10
solver = CrankNicolsonRK4(inner_steps=10)


def compute_loss(net, u0, yy):
    u_f, u_a = jax.vmap(solver.unroll, (None, 0, 0))(net, u0, yy)
    loss_4dvar = ((u_a[:, 0] - yy[:, 0]) ** 2).mean() + (
        (u_f[:, 1:] - yy[:, 1:]) ** 2
    ).mean()
    return loss_4dvar


def train(
    fname: str,
    lr0: float = 1e-3,
    epoch: int = 100000,
    noise_level: int = 0,
    rank: int = 32,
):
    net = ConvNet2d(rank=rank, kernel_size=10)
    lr = optax.cosine_decay_schedule(lr0, epoch)
    opt = optax.lion(lr)
    solver = jaxopt.OptaxSolver(compute_loss, opt)
    solver_step = jax.jit(solver.update)

    # train
    data_loader = DataLoader(noise_level)
    _, u0, _, yy = data_loader.load_train(unroll_length=10)
    state = solver.init_state(net, u0, yy)
    net, state, loss_traj = solve(
        solver_step, net, state, u0[:100], yy[:100], maxiter=epoch
    )

    # save checkpoint
    eqx.tree_serialise_leaves(f"results/{fname}.eqx", net)
    return net, loss_traj


def test_on(set: str, noise_level, net, unroll_length: int = 60):
    data_loader = DataLoader(noise_level)
    tt, u0, uu_ref, yy = data_loader.load_test(f"data/{set}.npz", unroll_length)
    uu_base = solver.solve(u0, tt)
    uu_f, uu_a = solver.unroll(net, u0, yy)
    uu = Solution(tt, uu_ref, uu_base, uu_f, uu_a, yy)
    return uu


def main(
    lr0: float = 1e-3,
    epoch: int = 200,
    noise_level: int = 75,
    rank: int = 64,
    include_training: bool = True,
):
    fname = f"kolmogorov_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)

    if include_training:
        net, loss_traj = train(
            fname, lr0=lr0, epoch=epoch, noise_level=noise_level, rank=rank
        )
    else:
        net = ConvNet2d(rank=rank, kernel_size=10)
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    # uu = test_on("train", noise_level, net, unroll_length=1000)

    uu = test_on("test", noise_level, net, unroll_length=5000)
    # uu.save(fname + "_test")
    # visualize(uu, loss_traj, fname=fname + "_test")

    print(f"""NRMSE.
          w/o assimilation: {np.linalg.norm(uu.baseline - uu.reference) / np.linalg.norm(uu.reference)}
          w/  assimilation: {np.linalg.norm(uu.forecast - uu.reference) / np.linalg.norm(uu.reference)}""")

    # transform the trajectory into real-space and wrap in xarray for plotting
    tt = uu.tt[1:]
    spatial_coord = np.arange(64) * 2 * np.pi / 64  # same for x and y
    coords = {
        "time": uu.tt[999::1000],
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
