import equinox as eqx
import numpy as np
import optax

from oda.models import Lorenz96
from oda.networks import ConvNet as Net
from oda.utils import DataLoader, Optimization, test_on, visualize
from oda.filters import ClassicFilter as Filter


def main(
    lr0: float = 1e-3,
    epoch: int = 200,
    noise_level: int = 0,
    rank: int = 32,
    include_training: bool = True,
    sensor_every: int = 1,
    test_unroll_length: int = 1000,
):
    fname = f"lorenz_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)
    model = Lorenz96(d_in=1, sensor_every=sensor_every)
    filter = Filter(model=model, observe=model.observe)
    net = Net(rank=rank, kernel_size=5, stride=sensor_every)
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"results/{fname}.eqx", net)
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

    print(f"""NRMSE.
          w/o assimilation: {np.linalg.norm(uu.baseline - uu.reference) / np.linalg.norm(uu.reference)}
          w/  assimilation: {np.linalg.norm(uu.forecast - uu.reference) / np.linalg.norm(uu.reference)}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
