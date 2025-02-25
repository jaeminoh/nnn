import os

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from tqdm import trange


def kursiv(Nx: int = 128, M: int = 16, draw_plot: bool = False):
    if not os.path.isdir("data"):
        os.mkdir("data")

    print(f"Precision check: {np.ones(()).dtype}")

    """
    A numerical solver of Kuramoto-Sivashinsky equation by ETDRK4 scheme
    """
    # spatial grid and initial condition
    xx = np.linspace(0, 32 * np.pi, Nx + 1)[:-1]
    u = np.cos(xx / 16) * (1 + np.sin(xx / 16))
    v = rfft(u)

    # precompute quantities
    t = 0.0
    h = 1 / 4
    k = 2j * np.pi * rfftfreq(Nx, 32 * np.pi / Nx)
    L = -(k**2) - k**4
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)
    M = 16  # the number of quadrature points for the contour integral
    r = np.exp(1j * np.pi * np.arange(1 - 0.5, M + 1 - 0.5) / M)
    LR = h * L[:, None] + r
    Q = h * ((np.exp(LR / 2) - 1) / LR).mean(1).real
    f1 = h * ((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(1).real
    f2 = h * ((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3).mean(1).real
    f3 = h * ((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3).mean(1).real

    vv = [v]
    tt = [0.0]
    tmax = 10000
    nmax = int(tmax // h)

    def step(t, v):
        t = t + h
        Nv = -0.5 * k * rfft(irfft(v, Nx) ** 2)
        a = E2 * v + Q * Nv
        Na = -0.5 * k * rfft(irfft(a, Nx) ** 2)
        b = E2 * v + Q * Na
        Nb = -0.5 * k * rfft(irfft(b, Nx) ** 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = -0.5 * k * rfft(irfft(c, Nx) ** 2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        return t, v

    for _ in trange(nmax):
        t, v = step(t, v)
        tt.append(t), vv.append(v)

    tt = np.stack(tt)
    vv = np.stack(vv)
    uu = irfft(vv, Nx, axis=-1)

    np.savez("data/train.npz", tt=tt[5000:-5000], sol=uu[5000:-5000])
    np.savez("data/test.npz", tt=tt[35000:], sol=uu[35000:])

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
