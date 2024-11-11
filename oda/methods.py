import jax
import jax.numpy as jnp
import numpy as np


class Euler:
    def __init__(self, problem):
        self.problem = problem

    def forecast(self, u0, dt=0.01):
        return u0 + dt * self.problem(u0)

    def solve(self, u0, tt):
        ulist = [u0]
        for dt in tt[1:] - tt[:-1]:
            ulist.append(self.forecast(ulist[-1], dt=dt))
        return np.stack(ulist[1:])

    @staticmethod
    def analysis(net, u_f, y, dt=0.01):
        return u_f + dt * net(u_f, y)

    def _scan_fn(self, net, u0, y, dt=0.01):
        u_f = self.forecast(u0, dt=dt)
        u_a = self.analysis(net, u_f, y, dt=dt)
        return u_a, jnp.stack([u_f, u_a])

    def unroll(self, net, u0, yy):
        _, out = jax.lax.scan(lambda u0, y: self._scan_fn(net, u0, y), u0, yy)
        return out[:, 0], out[:, 1]  # u_f, u_a
