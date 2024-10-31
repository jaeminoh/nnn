from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray


class MultiLayerPerceptron(eqx.Module):
    layers: list

    def __init__(
        self,
        *,
        d_in: Union[str, int] = 2,
        width: int = 32,
        depth: int = 4,
        d_out: Union[str, int] = "scalar",
        key: PRNGKeyArray = jr.key(4321),
    ):
        layers = [d_in] + [width for _ in range(depth - 1)] + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]

    def __call__(self, *inputs):
        x = jnp.hstack(inputs)
        for layer in self.layers[:-1]:
            x = jax.nn.swish(layer(x))
        return jax.nn.gelu(self.layers[-1](x))


class TensorNet(eqx.Module):
    net_u: eqx.nn.Conv1d
    net_y: eqx.nn.Conv1d
    linear: eqx.nn.Linear

    def __init__(
        self,
        *,
        d_in: Union[str, int] = 2,
        d_out: Union[str, int] = "scalar",
        rank: int = 32,
        key: PRNGKeyArray = jr.key(4321),
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.net_u = eqx.nn.Conv1d(
            in_channels=d_in,
            out_channels=rank,
            kernel_size=4,
            stride=1,
            padding="SAME",
            padding_mode="circular",
            key=key1,
        )
        self.net_y = eqx.nn.Conv1d(
            in_channels=d_in,
            out_channels=rank,
            kernel_size=4,
            stride=1,
            padding="SAME",
            padding_mode="circular",
            key=key2,
        )
        self.linear = eqx.nn.Linear(rank, d_out, key=key3)

    def __call__(self, u, y):
        net_u = jax.nn.swish(self.net_u(u[:, None])).squeeze()
        net_y = jax.nn.swish(self.net_y(y[:, None])).squeeze()
        return jax.nn.swish(self.linear(net_u * net_y))


class Siren(eqx.Module):
    layers: list
    w0: jnp.ndarray  # adpative activation function

    def __init__(
        self,
        *,
        d_in: int = 3,
        width: int = 64,
        depth: int = 3,
        d_out: int = 2,
        w0: float = 10.0,
        key=jr.key(4321),
    ):
        layers = [d_in] + [width for _ in range(depth - 1)] + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(in_, out_, key=key)
            for in_, out_, key in zip(layers[:-1], layers[1:], keys)
        ]
        self.w0 = jnp.array(w0)
        self = siren_init(self, key)

    def __call__(self, *args):
        x = jnp.hstack(args)
        for layer in self.layers[:-1]:
            x = jnp.sin(self.w0 * layer(x))
        return self.layers[-1](x)


def siren_init(siren: Siren, key: PRNGKeyArray = jr.key(4123)):
    def init_weight(layer: eqx.nn.Linear, is_first: bool, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        if is_first:
            scale = 1 / d_in
        else:
            scale = jnp.sqrt(6 / d_in) / siren.w0
        W = jr.uniform(key, (d_out, d_in), minval=-1, maxval=1) * scale
        return W

    def init_bias(layer: eqx.nn.Linear, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        scale = jnp.sqrt(1 / d_in)
        b = jr.uniform(key, (d_out,), minval=-1, maxval=1) * scale
        return b

    num_layers = len(siren.layers)
    is_first = [True] + [False for _ in range(num_layers - 1)]
    keys = jr.split(key, num_layers)

    def get_weights(siren):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.weight for x in jtu.tree_leaves(siren, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    def get_biases(siren):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.bias for x in jtu.tree_leaves(siren, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    new_weight = list(map(init_weight, siren.layers, is_first, keys))
    new_bias = list(map(init_bias, siren.layers, keys))

    siren = eqx.tree_at(get_weights, siren, new_weight)
    siren = eqx.tree_at(get_biases, siren, new_bias)

    return siren
