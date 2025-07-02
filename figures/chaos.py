import diffrax as dfx
import jax.numpy as jnp
import jax
import numpy as np

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
    ts = np.arange(0, 1.0, 0.01)
    saveat = dfx.SaveAt(ts=ts)

    # ode problem
    lorenz96 = Lorenz96(Nx=Nx)
    prob = dfx.ODETerm(lambda t, u, args: lorenz96(u))

    # solve!
    us = solve(prob, u0, saveat)
    vs = solve(prob, v0, saveat)
    
    # error
    es = np.linalg.norm(us - vs, axis=-1)


    # save
    np.savez("data/chaos.npz", us=us[np.array([0, -1])], vs=vs[np.array([0, -1])], error=es)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
