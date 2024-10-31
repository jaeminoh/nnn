import jax
import jax.numpy as jnp
import numpy as np

from oda.problems import lorenz96


class Euler:
    def __init__(self):
        self.forecast = euler_forecast
        self.analysis = euler_analysis
        self.solve = euler_solve
        self.unroll = unroll


def euler_forecast(u0, dt=0.01):
    return u0 + dt * lorenz96(u0)


def euler_solve(u0, tt):
    ulist = [u0]
    for dt in tt[1:] - tt[:-1]:
        ulist.append(euler_forecast(ulist[-1], dt=dt))
    return np.stack(ulist[1:])


def euler_analysis(net, u_f, y, dt=0.01):
    return u_f + dt * net(u_f, y)


def _scan_fn(net, u0, y, dt=0.01):
    u_f = euler_forecast(u0, dt=dt)
    u_a = euler_analysis(net, u_f, y, dt=dt)
    return u_a, jnp.stack([u_f, u_a])


def unroll(net, u0, yy):
    _, out = jax.lax.scan(lambda u0, y: _scan_fn(net, u0, y), u0, yy)
    return out[:, 0], out[:, 1]  # u_f, u_a
