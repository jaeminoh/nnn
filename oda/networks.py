import equinox as eqx
import jax
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, PRNGKeyArray, jaxtyped


class ConvNet(eqx.Module):
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
        rank: int = 32,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2 = jr.split(key)
        self.encoder = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dim,
            in_channels=1,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key1,
        )
        self.decoder = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=rank,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key2,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        u: Float[ArrayLike, "*Nx"],
        Hu: Float[ArrayLike, "*No"],
        y: Float[ArrayLike, "*No"],
    ):
        return self.decoder(jax.nn.swish(self.encoder((Hu - y)[None, ...]))).squeeze()


class ConvOperator(eqx.Module):
    encoder: eqx.nn.Conv
    decoder: eqx.nn.ConvTranspose
    innovation: ConvNet

    def __init__(
        self,
        *,
        num_spatial_dim: int = 1,
        rank: int = 32,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.encoder = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dim,
            in_channels=1,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key1,
        )
        self.decoder = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=rank,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="CIRCULAR",
            key=key2,
        )
        self.innovation = ConvNet(
            num_spatial_dim=num_spatial_dim,
            rank=rank,
            kernel_size=kernel_size,
            stride=stride,
            key=key,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        u: Float[ArrayLike, "*Nx"],
        Hu: Float[ArrayLike, "*No"],
        y: Float[ArrayLike, "*No"],
    ) -> Float[ArrayLike, "*Nx"]:
        branch = self.innovation(u, Hu, y)
        trunk = self.decoder(jax.nn.swish(self.encoder(u[None, ...]))).squeeze()
        return branch * trunk


class Observation(eqx.Module):
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
    ) -> Float[ArrayLike, "*Ny"]:
        return self.conv(jax.nn.swish(self.downsampling(u[None, ...]))).squeeze()


class ObservationTranspose(eqx.Module):
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
