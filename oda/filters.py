import jax
import jax.numpy as jnp

from oda.models import DynamicalCore


class Filter:
    def __init__(self, model: DynamicalCore):
        self.filtering_scale = model.dt * model.inner_steps

    def analysis(self, net, u_f, y):
        """
        Neural filtering (or correction) of forecast `u_f` based on observation `y`.
        Returns analysis `u_a`.
        """
        return u_f + self.filtering_scale * net(self.observe(u_f), y)

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
