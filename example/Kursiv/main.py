import equinox as eqx
import numpy as np
import optax

from oda.filters import ClassicFilter as Filter
from oda.models import Kursiv
from oda.networks import DNO, LinearCorrector
from oda.utils import DataLoader, Optimization, rmse, test_on, visualize

def main(
        filter_type: str = "nonlinear",
        noise_level: float = 0.5,
        sensor_every: int = 1,
        rank: int = 20,
        lr0: float = 1e-3,
        epoch: int = 300,
        include_training: bool = True,
        unroll_length: int = 10,
        inner_steps: int = 10,
        method: str = "forward_euler",
):
    fname = f"Kursiv_{filter_type}_Noise{noise_level}Obs{sensor_every}Rank{rank}"
    print(f"""
          ============================
          Configurations:
          filter_type: {filter_type}
          method: {method}
          inner_steps: {inner_steps}
          noise_level: {noise_level}
          sensor_every: {sensor_every}
          rank: {rank}
          lr0: {lr0}
          epoch: {epoch}
          include_training: {include_training}
          unroll_length: {unroll_length}
          """)
    model = Kursiv(sensor_every=sensor_every, d_in=1, method=method, inner_steps=inner_steps)
    filter = Filter(model=model, observe=model.observe)
    if filter_type == "nonlinear":
        net = DNO(
            stride=sensor_every,
            Nx=128,
            num_channels=rank,
            num_spatial_dims=1,
        )
    elif filter_type == "linear":
        net = LinearCorrector(
            d_in=1,
            Nx=128,
            sensor_every=sensor_every,
        )
    else:
        raise ValueError(f"Unknown filter type! It should be 'nonlinear' or 'linear', but got {filter_type}.")
    
    data_loader = DataLoader(model.observe, noise_level=noise_level)

    if include_training:
        opt = Optimization(lr0=lr0, algorithm=optax.adam, epoch=epoch)
        train_data = data_loader.load_train(unroll_length=10)
        net, loss_traj = opt.solve(fname, filter, net, train_data)
    else:
        net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)
        loss_traj = np.ones((epoch // 100,))

    uu = test_on("train", filter, net, data_loader=data_loader, unroll_length=1000)
    visualize(uu, loss_traj, fname=fname + "_train")

    # Test on test sets with different unroll lengths
    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=1000)
    uu.save(fname + "_test1")
    visualize(uu, loss_traj, fname=fname + "_test1")

    uu = test_on("test", filter, net, data_loader=data_loader, unroll_length=5000)
    uu.save(fname + "_test2")
    visualize(uu, loss_traj, fname=fname + "_test2")

    print(f"""
          RMSE:  {rmse(uu.forecast, uu.reference)}
          nRMSE: {rmse(uu.forecast, uu.reference, normalize=True)}
          =============================""")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
