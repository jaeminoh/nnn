import os

import diffrax as dfx
import jax
import numpy as np

from nnn.equations import Lorenz96


def main(Nx: int = 40, draw_plot: bool = False, forcing: float = 8):
    """
    A numerical solver of Lorenz 96 model
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    path = "data/Lorenz96"
    if not os.path.isdir(path):
        os.makedirs(path)
    print(f"""Generate L96 data.
          check precision: {jax.numpy.ones(()).dtype}
          Nx: {Nx}, draw plot: {draw_plot}, forcing: {forcing}""")

    # initial condition
    u0 = np.ones((Nx,)) * forcing
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

    # solve RK4
    def solve_rk4(u0):
        uu = [u0]
        for i in range(1, (len(tt) - 1) * 15 + 1):
            u0 = lorenz96._step(u0)
            if i % 15 == 0:
                uu.append(u0)
        return np.stack(uu)

    uu = solve_rk4(u0)

    np.savez(f"{path}/train.npz", tt=tt[80:-400], sol=uu[80:-400])
    np.savez(f"{path}/test.npz", tt=tt[-401:], sol=uu[-401:])
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
        plt.savefig(f"{path}/lorenz96.pdf", format="pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
