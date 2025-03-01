import equinox as eqx
import numpy as np
import optax

from oda.filters import ClassicFilter as Filter
from oda.models import Lorenz96
from oda.networks import MultiLayerPerceptron as Net
from oda.utils import DataLoader, Optimization, test_on, visualize, rmse


def main(
    lr0: float = 1e-3,
    epoch: int = 200,
    noise_level: int = 36,
    rank: int = 32,
    include_training: bool = True,
    sensor_every: int = 1,
    test_unroll_length: int = 100,
    Nx: int = 128,
):
    fname = f"lorenz_lr{lr0}_epoch{epoch}_noise{noise_level}_rank{rank}"
    print(fname)
    model = Lorenz96(d_in=1, Nx=Nx, sensor_every=sensor_every, inner_steps=10)
    filter = Filter(model=model, observe=model.observe)
    # net = Net(hidden_channels=rank, kernel_size=5, stride=sensor_every, num_spatial_dim=1)
    net = Net(d_in=Nx // sensor_every, width=rank, depth=2, d_out=Nx, w0=5.0)
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.lion, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10)
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

    norm_reference = np.linalg.norm(uu.reference)
    norm_reference = 1

    print(f"""RMSE.
          w/o assimilation: {rmse(uu.baseline - uu.reference) / norm_reference}
          w/  assimilation: {rmse(uu.forecast - uu.reference) / norm_reference}""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
