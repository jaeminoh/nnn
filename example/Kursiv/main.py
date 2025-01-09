import equinox as eqx
import numpy as np
import optax

from oda.filters import ClassicFilter as Filter
from oda.models import Kursiv
from oda.networks import SimpleCorrector as Net
from oda.utils import DataLoader, Optimization, rmse, test_on, visualize


def main(
    lr0: float = 1e-3,
    epoch: int = 100000,
    noise_level: int = 0,
    rank: int = 32,
    include_training: bool = True,
    sensor_every: int = 1,
):
    fname = f"kursiv_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)
    model = Kursiv(sensor_every=sensor_every, d_in=1)
    filter = Filter(model=model, observe=model.observe)
    net = Net(
        hidden_channels=rank, kernel_size=5, stride=sensor_every, num_spatial_dim=1
    )
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("train", filter, net, data_loader=data_loader, unroll_length=1000)
    visualize(uu, loss_traj, fname=fname + "_train")

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=1000)
    uu.save(fname + "_test")
    visualize(uu, loss_traj, fname=fname + "_test")

    print(f"""RMSE.
          w/o assimilation: {rmse(uu.baseline - uu.reference)}
          w/  assimilation: {rmse(uu.forecast - uu.reference)}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
