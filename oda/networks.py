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
        # self = convert_mlp_to_siren(self)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, Hu: Float[ArrayLike, " No"], y: Float[ArrayLike, " No"]):
        x = Hu - y
        for layer in self.layers[:-1]:
            x = jnp.tanh(self.w0 * layer(x))
        return self.layers[-1](x).squeeze()


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
