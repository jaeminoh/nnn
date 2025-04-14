import equinox as eqx
import numpy as np
import optax

from oda.filters import ClassicFilter as Filter
from oda.models import Kursiv
from oda.networks import SimpleCorrector as Net
from oda.utils import DataLoader, Optimization, rmse, test_on, visualize


def main(
    lr0: float = 1e-3,
    epoch: int = 1000,
    noise_level: int = 1,  # 0 is nonsense, since exact initial condition.
    rank: int = 16,
    include_training: bool = True,
    sensor_every: int = 4,
):
    fname = f"kursiv_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)
    model = Kursiv(sensor_every=sensor_every, d_in=1)
    filter = Filter(model=model, observe=model.observe)
    net = Net(
        kernel_size=5,
        stride=sensor_every,
        num_spatial_dim=1,
        hidden_channels=rank,
    )
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("train", filter, net, data_loader=data_loader, unroll_length=1000)
    visualize(uu, loss_traj, fname=fname + "_train")

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=1000)
    uu.save(fname + "_test1")
    visualize(uu, loss_traj, fname=fname + "_test1")

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=5000)
    uu.save(fname + "_test2")
    visualize(uu, loss_traj, fname=fname + "_test2")

    print(f"""RMSE.
          w/o assimilation: {rmse(uu.baseline - uu.reference)}
          w/  assimilation: {rmse(uu.forecast - uu.reference)}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
