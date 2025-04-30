import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, PRNGKeyArray, jaxtyped
from typing import Union
import jax.tree_util as jtu


class BaseCorrector(eqx.Module):
    def __call__(self, u_like, y_like):
        raise NotImplementedError


class SimpleCorrector(BaseCorrector):
    """
    Circular padding if periodic spatial domain.

    net = (Hu, y) -> Conv -> activation -> ConvTransposed
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


class MultiLayerPerceptron(eqx.Module):
    layers: list
    w0: jnp.ndarray = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_in: Union[str, int] = 2,
        width: int = 32,
        depth: int = 4,
        d_out: Union[str, int] = "scalar",
        key: PRNGKeyArray = jr.key(4321),
        w0: float = 10.0,
    ):
        layers = [d_in] + [width] * (depth - 1) + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.w0 = jnp.array(w0)
        self = convert_mlp_to_siren(self)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.sin(self.w0 * layer(x))
        return self.layers[-1](x).squeeze()


def siren_init(mlp: MultiLayerPerceptron, key: PRNGKeyArray = jr.key(4123)):
    def init_weight(layer: eqx.nn.Linear, is_first: bool, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        if is_first:
            scale = 1 / d_in
        else:
            scale = jnp.sqrt(6 / d_in) / mlp.w0
        W = jr.uniform(key, (d_out, d_in), minval=-1, maxval=1) * scale
        return W

    def init_bias(layer: eqx.nn.Linear, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        scale = jnp.sqrt(1 / d_in)
        b = jr.uniform(key, (d_out,), minval=-1, maxval=1) * scale
        return b

    num_layers = len(mlp.layers)
    is_first = [True] + [False for _ in range(num_layers - 1)]
    keys = jr.split(key, num_layers)

    def get_weights(mlp: MultiLayerPerceptron):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.weight for x in jtu.tree_leaves(mlp, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    def get_biases(mlp: MultiLayerPerceptron):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.bias for x in jtu.tree_leaves(mlp, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    new_weight = list(map(init_weight, mlp.layers, is_first, keys))
    new_bias = list(map(init_bias, mlp.layers, keys))

    mlp = eqx.tree_at(get_weights, mlp, new_weight)
    mlp = eqx.tree_at(get_biases, mlp, new_bias)

    return mlp


def convert_mlp_to_siren(net: eqx.Module, key=jr.key(4321)):
    def is_mlp(mlp: MultiLayerPerceptron):
        return isinstance(mlp, MultiLayerPerceptron)

    def get_mlps(net: eqx.Module):
        return [x for x in jtu.tree_leaves(net, is_mlp) if is_mlp(x)]

    mlps = get_mlps(net)
    num_mlps = len(mlps)
    keys = jr.split(key, num_mlps)
    new_mlps = list(map(siren_init, mlps, keys))

    net = eqx.tree_at(get_mlps, net, new_mlps)
    return net


class AutoEncoder(eqx.Module):
    encoders: list
    decoders: list

    def __init__(
        self,
        *,
        num_channels: int = 40,
        kernel_size: int = 4,
        stride: int = 1,
        key: PRNGKeyArray = jr.key(4321),
        num_spatial_dim: int = 1,
    ):
        key1, key2 = jr.split(key)
        hidden_channels_list = [1] + [num_channels] * 3
        strides_list = [1, 1, stride]
        self.encoders = [
            eqx.nn.ConvTranspose(
                num_spatial_dim,
                _in,
                _out,
                kernel_size,
                _s,
                padding="SAME",
                padding_mode="ZEROS",
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
            x = jax.nn.swish(encoder(x))
        for decoder in self.decoders[:-1]:
            x = jax.nn.swish(decoder(x))
        return self.decoders[-1](x).squeeze()


class DNO(eqx.Module):
    branch: MultiLayerPerceptron
    trunk: AutoEncoder

    def __init__(self, stride: int = 2, Nx: int = 40, num_channels: int = 40, key = jr.key(4321)):
        key_b, key_t = jr.split(key)
        self.branch = MultiLayerPerceptron(d_in=Nx, d_out=num_channels, key=key_b)
        self.trunk = AutoEncoder(
            num_channels=num_channels,
            kernel_size=3,
            stride=stride,
            key=key_t,
        )
    
    @jaxtyped(typechecker=typechecker)
    def __call__(self, u: Float[ArrayLike, "*Nx"], y: Float[ArrayLike, "*No"]) -> Float[ArrayLike, "*Nx"]:
        branch = self.branch(u) # (num_channels,)
        trunk = self.trunk(y) # num_channels x (No * stride)?
        return branch @ trunk # (No * stride,)


if __name__ == "__main__":
    import numpy as np
    nnn = DNO()
    print(nnn(np.ones((40,)), np.zeros((20,))).shape) # should be (40,).