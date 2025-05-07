import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, jaxtyped

from oda.models import DynamicalCore
from oda.observation import ObservationOperator


class BaseFilter:
    def __init__(self, model: DynamicalCore, observe: ObservationOperator):
        self.model = model
        self.observe = observe

    def analysis(self):
        """
        Neural filtering (or correction) of forecast `u_f` based on observation `y`.
        Returns analysis `u_a`.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _compute_loss(
        self, net, u0: Float[ArrayLike, " *Nx"], yy: Float[ArrayLike, " Nt *No"]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        """
        Loss function motivated by the 4DVAR method.

        **args**
        - net: neural network
        - u0: initial condition
        - yy: observations

        **returns**
        - j0: fit analysis and observation
        - j1: fit forecast and observation
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _scan_fn(self, net, u0, y):
        u_f = self.model.forecast(u0)
        u_a = self.analysis(net, u_f, y)
        return u_a, jnp.stack([u_f, u_a])

    @jaxtyped(typechecker=typechecker)
    def unroll(
        self, net, u0: Float[ArrayLike, "*Nx"], yy: Float[ArrayLike, " Nt *No"]
    ) -> tuple[Float[ArrayLike, " Nt *Nx"], Float[ArrayLike, " Nt *Nx"]]:
        """
        Fast (differentiable) for-loop for forecast and analysis.
        Returns `u_f` and `u_a`.

        The number of iterations, the number of rows of `out` and `yy` are the same.
        """
        _, out = jax.lax.scan(lambda u0, y: self._scan_fn(net, u0, y), u0, yy)
        return out[:, 0], out[:, 1]  # u_f, u_a

    @jaxtyped(typechecker=typechecker)
    def compute_loss(
        self, net, u0: Float[ArrayLike, " Ne *Nx"], uu: Float[ArrayLike, " Ne Nt *Nx"], yy: Float[ArrayLike, " Ne Nt *No"]
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
        #return (j0**2).mean() + (j1**2).mean()
        return (j0**2).mean()


class ClassicFilter(BaseFilter):
    def __init__(self, loss_type: int = "supervised", **basefilter_kwargs):
        super().__init__(**basefilter_kwargs)
        if loss_type == "self_supervised":
            self._compute_loss = self._self_supervised_loss
        elif loss_type == "supervised":
            self._compute_loss = self._supervised_loss

    def analysis(self, net, u_f, y):  # first order operator splitting
        #return u_f + net(self.observe(u_f), y) * self.model.dt * self.model.inner_steps
        return u_f + net(u_f, y - self.observe(u_f)) * self.model.dt * self.model.inner_steps
    
    def _self_supervised_loss(self, net,
                              u0: Float[ArrayLike, " *Nx"],
                              uu: Float[ArrayLike, " Nt ..."],
                              yy: Float[ArrayLike, " Nt ..."]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = self.observe(u_a[0]) - yy[0]
        j1 = jax.vmap(self.observe)(u_f[1:]) - yy[1:]
        return j0, j1
    
    def _supervised_loss(self,
                 net,
                 u0: Float[ArrayLike, " *Nx"],
                 uu: Float[ArrayLike, " Nt ..."],
                 yy: Float[ArrayLike, " Nt ..."]
                 ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = u_a[0] - uu[0]
        j1 = u_f[1:] - uu[1:]
        return j0, j1



    def _compute_loss(
        self, net, u0: Float[ArrayLike, " *Nx"], yy: Float[ArrayLike, " Nt ..."]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = self.observe(u_a[0]) - yy[0]
        j1 = jax.vmap(self.observe)(u_f[1:]) - yy[1:]
        tt = jnp.arange(yy.shape[0] - 1)
        return j0, j1 * tt[:, None]


##########
# Unused #
##########
class LearnableObservationFilter(BaseFilter):
    def __init__(self, **basefilter_kwargs):
        super().__init__(**basefilter_kwargs)

    def analysis(self, net, u_f, y):
        return u_f + net(u_f, y) * self.model.dt * self.model.inner_steps

    def _compute_loss(
        self, net, u0: Float[ArrayLike, " *Nx"], yy: Float[ArrayLike, " Nt ..."]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = net.u_to_y(u_a[0]) - yy[0]
        j1 = jax.vmap(net.u_to_y)(u_f[1:]) - yy[1:]
        return j0, j1


class ObservationTransposeFilter(BaseFilter):
    def __init__(self, **basefilter_kwargs):
        super().__init__(**basefilter_kwargs)

    def analysis(self, net, u_f, y):
        return u_f + net(u_f, y) * self.model.dt * self.model.inner_steps

    def _compute_loss(
        self, net, u0: Float[ArrayLike, " *Nx"], yy: Float[ArrayLike, " Nt ..."]
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, " Nt-1 ..."]]:
        u_f, u_a = self.unroll(net, u0, yy)
        j0 = u_a[0] - net.y_to_u(yy[0])
        j1 = u_f[1:] - jax.vmap(net.y_to_u)(yy[1:])
        return j0, j1
