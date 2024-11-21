from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class Euler:
    def __init__(self, problem, dt: float = 0.01, num_steps: int = 1):
        self.problem = problem
        self.dt = dt
        self.num_steps = num_steps

    @partial(jax.jit, static_argnums=0)
    def _step(self, u0):
        return u0 + self.dt * self.problem(u0)

    def forecast(self, u0):
        "Repeated application of the forward Euler step (`_step`) for `self.num_steps` times."
        for _ in range(self.num_steps):
            u0 = self._step(u0)
        return u0

    def solve(self, u0, tt):
        "Solve initial value problem following the time discretization `tt`."
        ulist = [u0]
        for _ in tt[1:] - tt[:-1]:
            ulist.append(self.forecast(ulist[-1]))
        return np.stack(ulist[1:])

    def analysis(self, net, u_f, y):
        """
        Neural filtering (or correction) of forecast `u_f` based on observation `y`.
        Returns analysis `u_a`.
        """
        return u_f + self.dt * self.num_steps * net(u_f, y)

    def _scan_fn(self, net, u0, y):
        u_f = self.forecast(u0)
        u_a = self.analysis(net, u_f, y)
        return u_a, jnp.stack([u_f, u_a])

    def unroll(self, net, u0, yy):
        """
        Fast (differentiable) for-loop for forecast and analysis.
        Returns `u_f` and `u_a`.

        The number of iterations, the number of rows of `out` and `yy` are the same.
        """
        _, out = jax.lax.scan(lambda u0, y: self._scan_fn(net, u0, y), u0, yy)
        return out[:, 0], out[:, 1]  # u_f, u_a
