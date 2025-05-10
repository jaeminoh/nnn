import os

import diffrax as dfx
import jax
import numpy as np

from oda.models import Lorenz96

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def main(Nx: int = 40, draw_plot: bool = False, forcing: int = 8):
    if not os.path.isdir("data"):
        print("no data directory, create one")
        os.mkdir("data")

    print(f"check precision: {jax.numpy.ones(()).dtype}")
    print(f"Nx: {Nx}, draw plot: {draw_plot}, forcing: {forcing}")
    """
    A numerical solver of Lorenz 96 model
    """
    # initial condition
    u0 = np.ones((Nx,)) * 8.0
    u0[0] += 0.01

    # time steps
    tt = np.arange(0, 315 + 0.15, 0.15)
    saveat = dfx.SaveAt(ts=tt)

    # ode problem
    lorenz96 = Lorenz96(Nx=Nx)
    prob = dfx.ODETerm(lambda t, u, args: lorenz96(u))

    # solve!
    def solve(u0):
        sol = dfx.diffeqsolve(
            prob,
            dfx.Dopri8(),
            t0=tt[0],
            t1=tt[-1],
            dt0=1e-2,
            y0=u0,
            saveat=saveat,
            stepsize_controller=dfx.PIDController(rtol=1e-9, atol=1e-9),
            max_steps=int(1e6),
        )
        uu = sol.ys
        return uu
    
    uu = solve(u0)

    np.savez("data/train.npz", tt=tt[80:-400], sol=uu[80:-400])
    np.savez("data/test.npz", tt=tt[-401:], sol=uu[-401:])
    print(f"train data shape: {uu[80:-400].shape}")
    print(f"test data shape: {uu[-401:].shape}")

    if draw_plot:
        import matplotlib.pyplot as plt

        plt.imshow(
            uu,
            aspect="auto",
            origin="lower",
            cmap="jet",
            extent=[0, 128, 0, 100],
        )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$t$")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("data/lorenz96.pdf", format="pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
