import diffrax as dfx
import jax
import numpy as np
import matplotlib.pyplot as plt

from oda.models import Lorenz96

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# --- LaTeX Configuration for Matplotlib ---
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
    "font.family": "serif",
    "font.size": 10,  # Adjust as needed
    "text.usetex": True,
    "pgf.rcfonts": False,  # Don't use pgf.rcfonts for custom fonts
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amsfonts}",
        r"\usepackage{amssymb}",
        r"\usepackage{newtxtext}",  # Example: Times New Roman clone for text
        r"\usepackage{newtxmath}",  # Example: Times New Roman clone for math
        # Add any other LaTeX packages you need for specific fonts or symbols
    ])
})
# ------------------------------------------


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
    uu = solve(prob, u0, saveat)[[0, -1]]
    vv = solve(prob, v0, saveat)[[0, -1]]
    np.savez("chaotic.npz", uu=uu, vv=vv)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
