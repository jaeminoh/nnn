import diffrax
import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

from oda.models import lorenz96


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, u, args):
        return self.mlp(u).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    func: Func

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        (key,) = jr.split(key, 1)
        self.func = Func(data_size, hidden_size, width_size, depth, key=key)

    def __call__(self, u0, ts, coeffs, unroll_out=False):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, u, args: lorenz96(u)),
            diffrax.ControlTerm(self.func, control),
        )
        if unroll_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Euler(),
            ts[0],
            ts[-1],
            0.01,
            u0,
            # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
        )
        return solution.ys


def main(length: int = 100, noise_level: int = 5):
    cde = NeuralCDE(128, 128, 128, 2, key=jr.key(4321))
    d = np.load("data/Tsit.npz")
    ts = d["tt"][: length + 1]
    us = d["sol"][: length + 1]
    ys = us + 0.01 * noise_level * jr.normal(jr.key(1234), us.shape)
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)

    @eqx.filter_jit
    def compute_loss(net: NeuralCDE):
        out = net(us[0], ts, coeffs, unroll_out=True)
        return ((out - us) ** 2).mean()

    grad_loss = eqx.filter_value_and_grad(compute_loss)
    opt = optax.lion(1e-3)
    opt_state = opt.init(eqx.filter(cde, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(net, opt_state):
        loss, grads = grad_loss(net)
        updates, opt_state = opt.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return loss, net, opt_state

    for it in (pbar:=trange(500)):
        loss, cde, opt_state = make_step(cde, opt_state)
        pbar.set_postfix({"Loss":f"{loss:.3e}"})

    # test
    test_length = length
    uu_base = np.load("data/Euler.npz")["sol"][: test_length + 1]
    ts = d["tt"][: test_length + 1]
    us = d["sol"][: test_length + 1]
    ys = us + 0.01 * noise_level * jr.normal(jr.key(2222), us.shape)
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    uu_p = cde(us[0], ts, coeffs, True)
    _, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    err = abs(us[1:] - uu_base[1:])
    err1 = abs(us[1:] - uu_p[1:])
    scale = abs(us[1:]).max()
    vmax = (np.stack([err, err1])).max()
    ax0.imshow(err, vmax=vmax, vmin=0, aspect="auto")
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$t$")
    ax0.set_title(f"Forward Euler, max: {err.max() / scale:.2e}")
    ax1.imshow(err1, vmax=vmax, vmin=0, aspect="auto")
    ax1.set_title(f"Posterior, max: {err1.max() / scale:.2e}")
    ax1.axis("off")
    ax2.plot(ts[1:], us[1:, 0], label="Reference", linewidth=3)
    ax2.plot(ts[1:], uu_base[1:, 0], "--", label="Baseline", linewidth=3)
    ax2.plot(ts[1:], uu_p[1:, 0], ":", label="Forecast", linewidth=3)
    ax2.set_title(f"noise level: {noise_level}% z")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$u(x_0)$")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(
        f"results/ncde_noise{noise_level}.png",
        dpi=300,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
