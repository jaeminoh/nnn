import diffrax as dfx
import jax
import numpy as np
import matplotlib.pyplot as plt

from oda.models import Lorenz96

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def solve(prob, y0, saveat):
    sol = dfx.diffeqsolve(
        prob,
        dfx.Dopri8(),
        t0=0.0,
        t1=1.0,
        dt0=1e-2,
        y0=y0,
        saveat=saveat,
        stepsize_controller=dfx.PIDController(rtol=1e-9, atol=1e-9),
        max_steps=int(1e6),
    )
    return sol.ys


def main(Nx: int = 40, draw_plot: bool = False):
    print(f"check precision: {jax.numpy.ones(()).dtype}")
    print(f"Nx: {Nx}, draw plot: {draw_plot}")
    """
    A numerical solver of Lorenz 96 model
    """
    # initial condition
    u0 = np.ones((Nx,)) * 8
    v0 = np.ones((Nx,)) * 8
    v0[0] += 0.001

    # time steps
    tt = np.arange(0, 1.0, 0.01)
    saveat = dfx.SaveAt(ts=tt)

    # ode problem
    lorenz96 = Lorenz96(Nx=Nx)
    prob = dfx.ODETerm(lambda t, u, args: lorenz96(u))

    # solve!
    uu = solve(prob, u0, saveat)
    vv = solve(prob, v0, saveat)

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
    ax0.plot(uu[0], label=r"$u_0$")
    ax0.plot(vv[0], label=r"$v_0$", linestyle="--")
    ax0.set_xlabel(r"$x$")
    ax0.legend()
    ax1.plot(uu[-1], label=r"$u_1$")
    ax1.plot(vv[-1], label=r"$v_1$", linestyle="--")
    ax1.set_xlabel(r"$x$")
    ax1.legend()
    plt.tight_layout()
    plt.savefig("chaotic.pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
