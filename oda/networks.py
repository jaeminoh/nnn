import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray


class ConvNet(eqx.Module):
    "Circular padding if periodic spatial domain."

    encoder: eqx.nn.Conv
    decoder: eqx.nn.ConvTranspose

    def __init__(
        self,
        *,
        num_spatial_dim: int = 1,
        rank: int = 32,
        kernel_size: int = 4,
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
            padding_mode="circular",
            key=key1,
        )
        self.decoder = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dim,
            in_channels=rank,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            padding_mode="circular",
            key=key2,
        )

    def __call__(self, u, y):
        return self.decoder(jax.nn.swish(self.encoder((u - y)[None, ...]))).squeeze()
