import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, Key, jaxtyped

from nnn._cores import NudgingTerm


class LinearTerm(NudgingTerm):
    linear: eqx.nn.Linear
    d_in: int = eqx.field(static=True)

    def __init__(self, d_in: int, Nx: int, sensor_every: int, key=jr.key(4321)):
        self.d_in = d_in
        No = Nx // sensor_every
        if d_in == 1:
            self.linear = eqx.nn.Linear(No, Nx, use_bias=False, key=key)
        elif d_in == 2:
            self.linear = eqx.nn.Linear(No**d_in, Nx**d_in, use_bias=False, key=key)

    def __call__(
        self, u: Float[ArrayLike, "*Nx"], innovation: Float[ArrayLike, "*No"]
    ) -> Float[ArrayLike, "*Nx"]:
        if self.d_in == 1:
            return self.linear(innovation)
        elif self.d_in == 2:
            out = self.linear((innovation).ravel())
            return out.reshape(u.shape)


class MultiLayerPerceptron(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        d_in: str | int = 2,
        width: int = 32,
        depth: int = 4,
        d_out: str | int = "scalar",
        key: Key = jr.key(4321),
    ):
        layers = [d_in] + [width] * (depth - 1) + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x).squeeze()


class AutoEncoder(eqx.Module):
    encoders: list[eqx.nn.ConvTranspose]
    decoders: list[eqx.nn.Conv]

    def __init__(
        self,
        *,
        num_channels: int = 40,
        kernel_size: int = 4,
        stride: int = 1,
        key: Key = jr.key(4321),
        num_spatial_dim: int = 1,
    ):
        key1, key2 = jr.split(key)
        hidden_channels_list = [1] + [num_channels] * 3
        strides_list = [stride]
        self.encoders = [
            eqx.nn.ConvTranspose(
                num_spatial_dim,
                _in,
                _out,
                kernel_size,
                _s,
                padding="SAME",
                padding_mode="CIRCULAR",
                key=_k,
            )
            for (_in, _out, _s, _k) in zip(
                hidden_channels_list[:-1],
                hidden_channels_list[1:],
                strides_list,
                jr.split(key1, len(hidden_channels_list) - 1),
            )
        ]

        self.decoders = [
            eqx.nn.Conv(
                num_spatial_dim,
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding="SAME",
                padding_mode="CIRCULAR",
                key=_k,
            )
            for _k in jr.split(key2, len(hidden_channels_list) - 1)
        ]

    def __call__(
        self,
        y: Float[ArrayLike, "*No"],
    ):
        x = y[None]
        for encoder in self.encoders:
            x = jax.nn.tanh(encoder(x))
        for decoder in self.decoders[:-1]:
            x = jax.nn.tanh(decoder(x))
        return self.decoders[-1](x).squeeze()


class NNNTerm(NudgingTerm):
    branch: MultiLayerPerceptron
    trunk: AutoEncoder

    def __init__(
        self,
        stride: int = 2,
        Nx: int = 40,
        num_channels: int = 40,
        num_spatial_dims: int = 1,
        key=jr.key(4321),
    ):
        key_b, key_t = jr.split(key)
        self.branch = MultiLayerPerceptron(
            d_in=Nx**num_spatial_dims, d_out=num_channels, key=key_b, depth=2, width=128
        )
        self.trunk = AutoEncoder(
            num_channels=num_channels,
            kernel_size=5,
            stride=stride,
            key=key_t,
            num_spatial_dim=num_spatial_dims,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, u: Float[ArrayLike, "*Nx"], innovation: Float[ArrayLike, "*No"]
    ) -> Float[ArrayLike, "*Nx"]:
        branch = self.branch(u.ravel())  # (num_channels,)
        trunk = self.trunk(innovation)  # num_channels x (No * stride)
        return jnp.einsum("i, i... -> ...", branch, trunk)


if __name__ == "__main__":
    nnn = NNNTerm()
    print(nnn(jnp.ones((40,)), jnp.zeros((20,))).shape)  # should be (40,).
