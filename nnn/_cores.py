import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from beartype import beartype
from jaxtyping import ArrayLike, Float, jaxtyped


class DynamicalCore:
    "Numerical implementation of a physical model"

    def __init__(
        self,
        *,
        Nx: int,
        dt: float,
        inner_steps: int,
    ):
        self.Nx = Nx
        self.dt = dt
        self.inner_steps = inner_steps
        self._step = jax.jit(self._step)  # compile

    def __call__(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def forecast(self, u0):
        "Repeated application of the `_step` for `self.inner_steps` times."
        for _ in range(self.inner_steps):
            u0 = self._step(u0)
        return u0

    def solve(self, u0, tt):
        "Solve initial value problem following the time discretization `tt`."
        ulist = [u0]
        for _ in tt[1:] - tt[:-1]:
            ulist.append(self.forecast(ulist[-1]))
        return np.stack(ulist[1:])


class ObservationOperator:
    pass


class NudgingTerm(eqx.Module):
    pass


class Filter:
    """
    This class defines how to combine
    1) Dynamical Core
    2) Observation Operator
    3) Nudging term
    to produce analyses.
    Also this class defines loss function to train the nudging term.
    """

    def __init__(
        self,
        model: DynamicalCore,
        observe: ObservationOperator,
        mean: ArrayLike | float = 0.0,
        std: ArrayLike | float = 1.0,
    ):
        self.model = model
        self.observe = observe
        self.mean = mean
        self.std = std

    def analysis(self, net, u_f, y):
        b_in = (u_f - self.mean) / self.std
        return (
            u_f
            + net(b_in, y - self.observe(u_f)) * self.model.dt * self.model.inner_steps
        )

    def _scan_fn(self, net: NudgingTerm, u0, y):
        u_f = self.model.forecast(u0)
        u_a = self.analysis(net, u_f, y)
        return u_a, jnp.stack([u_f, u_a])

    @jaxtyped(typechecker=beartype)
    def unroll(
        self,
        net: NudgingTerm,
        u0: Float[ArrayLike, "*Nx"],
        yy: Float[ArrayLike, " Nt *No"],
    ) -> tuple[Float[ArrayLike, " Nt *Nx"], Float[ArrayLike, " Nt *Nx"]]:
        """
        Fast (differentiable) for-loop for forecast and analysis.
        Returns `u_f` and `u_a`.

        The number of iterations, the number of rows of `out` and `yy` are the same.
        """
        _, out = jax.lax.scan(lambda u0, y: self._scan_fn(net, u0, y), u0, yy)
        return out[:, 0], out[:, 1]

    @jaxtyped(typechecker=beartype)
    def _compute_loss(
        self,
        net: NudgingTerm,
        u0: Float[ArrayLike, " *Nx"],
        uu: Float[ArrayLike, " Nt ..."],
        yy: Float[ArrayLike, " Nt ..."],
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = u_a[0] - uu[0]
        j1 = u_f[1:] - uu[1:]
        # tt = jnp.arange(yy.shape[0] - 1)
        return j0, j1

    @jaxtyped(typechecker=beartype)
    def compute_loss(
        self,
        net: NudgingTerm,
        u0: Float[ArrayLike, " Ne *Nx"],
        uu: Float[ArrayLike, " Ne Nt *Nx"],
        yy: Float[ArrayLike, " Ne Nt *No"],
    ) -> Float[ArrayLike, ""]:
        """
        Loss function motivated by the 4DVAR method.

        **args**
        - net: neural network
        - u0: initial conditions of an ensemble
        - yy: observations of an ensemble

        **returns**
        - loss: mean squared error of the fit
        """
        j0, j1 = jax.vmap(lambda u0, uu, yy: self._compute_loss(net, u0, uu, yy))(
            u0, uu, yy
        )
        return (j0**2).mean() + (j1**2).mean()
