import os

import jax
import numpy as np
from tqdm import trange

from nnn.models import Kursiv

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def kursiv(Nx: int = 128, tmax: float = 1e4, draw_plot: bool = False):
    if not os.path.isdir("data"):
        os.mkdir("data")

    
    print(f"Data precision: {jax.numpy.ones(()).dtype}")

    """
    A numerical solver of Kuramoto-Sivashinsky equation by ETDRK4 scheme
    """
    # spatial grid and initial condition
    xx = np.linspace(0, 32 * np.pi, Nx + 1)[:-1]
    u = np.cos(xx / 16) * (1 + np.sin(xx / 16))
    model = Kursiv(Nx=Nx, method="etdrk4", inner_steps=1)
    print(f"dt = {model.dt}")

    t = 0.0

    uu = [u]
    tt = [0.0]
    nmax = int(tmax // model.dt)

    for _ in trange(nmax):
        t += model.dt
        u = model._step(u)
        tt.append(t), uu.append(u)

    tt = np.stack(tt)
    uu = np.stack(uu)

    if tmax == 1e4:
        np.savez("data/train.npz", tt=tt[5000:-25000], sol=uu[5000:-25000])
        np.savez("data/test.npz", tt=tt[15000:], sol=uu[15000:])
    else:
        np.savez("data/kursiv.npz", tt=tt, sol=uu)

    if draw_plot:
        import matplotlib.pyplot as plt

        plt.imshow(
            uu,
            aspect="auto",
            origin="lower",
            cmap="jet",
            extent=[0, 32 * np.pi, 0, tmax],
        )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$t$")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("data/kursiv.pdf", format="pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(kursiv)
