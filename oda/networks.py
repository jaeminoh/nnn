import equinox as eqx
import jax
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, PRNGKeyArray, jaxtyped


class BaseCorrector(eqx.Module):
    def __call__(self, u_like, y_like):
        raise NotImplementedError


class _StateToObservation(eqx.Module):
    downsampling: eqx.nn.Conv
    conv: eqx.nn.Conv

    def __init__(
        self,
        *,
        num_spatial_dim: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2 = jr.split(key)
        self.downsampling = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dim,
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key1,
        )
        self.conv = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key2,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        u: Float[ArrayLike, "*Nx"],
    ) -> Float[ArrayLike, "*No"]:
        return self.conv(jax.nn.swish(self.downsampling(u[None, ...]))).squeeze()


class SimpleCorrector(BaseCorrector):
    """
    Circular padding if periodic spatial domain.

    net = input -> Conv -> activation -> ConvTransposed
    """

    encoder: eqx.nn.Conv
    decoder: eqx.nn.ConvTranspose

    def __init__(
        self,
        *,
        num_spatial_dim: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2 = jr.split(key)
        self.encoder = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            padding_mode="ZEROS",
            key=key1,
        )
        self.decoder = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dim,
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key2,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        Hu: Float[ArrayLike, "*No"],
        y: Float[ArrayLike, "*No"],
    ):
        return self.decoder(jax.nn.swish(self.encoder((Hu - y)[None, ...]))).squeeze()


class LearnableObservationCorrector(BaseCorrector):
    u_to_y: _StateToObservation
    corrector: SimpleCorrector

    def __init__(self, *, key: PRNGKeyArray = jr.key(4321), **cnn_kwargs):
        key1, key2 = jr.split(key)
        self.u_to_y = _StateToObservation(**cnn_kwargs, key=key1)
        self.corrector = SimpleCorrector(**cnn_kwargs, key=key2)

    def __call__(self, u_f, y):
        return self.corrector(self.u_to_y(u_f), y)


class _ObservationToState(eqx.Module):
    upsampling: eqx.nn.ConvTranspose
    conv: eqx.nn.Conv

    def __init__(
        self,
        *,
        num_spatial_dim: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2 = jr.split(key)
        self.upsampling = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key1,
        )
        self.conv = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dim,
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key2,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        y: Float[ArrayLike, "*No"],
    ) -> Float[ArrayLike, "*Nx"]:
        return self.conv(jax.nn.swish(self.upsampling(y[None, ...]))).squeeze()


class ObservationTransposeCorrector(BaseCorrector):
    y_to_u: _ObservationToState
    corrector: SimpleCorrector

    def __init__(self, *, key: PRNGKeyArray = jr.key(4321), **cnn_kwargs):
        key1, key2 = jr.split(key)
        self.y_to_u = _ObservationToState(**cnn_kwargs, key=key1)
        self.corrector = SimpleCorrector(
            num_spatial_dim=cnn_kwargs["num_spatial_dim"],
            hidden_channels=cnn_kwargs["hidden_channels"],
            kernel_size=cnn_kwargs["kernel_size"],
            key=key2,
        )

    def __call__(
        self, u: Float[ArrayLike, "*Nx"], y: Float[ArrayLike, "*No"]
    ) -> Float[ArrayLike, "*Nx"]:
        return self.corrector(u, self.y_to_u(y))
