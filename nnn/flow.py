import equinox as eqx
import jax.random as jr
import jax
import jax.numpy as jnp
import optax
import jaxopt
from tqdm import trange


class Flow(eqx.Module):
    layers: list

    def __init__(self, dim: int = 2, h: int = 128, *, key=jr.key(777)):
        self.layers = (
            [eqx.nn.Linear(dim + 1, h, key=key)]
            + [eqx.nn.Linear(h, h, key=key)] * 3
            + [eqx.nn.Linear(h, dim, key=key)]
        )

    def __call__(self, t, x):
        x = jnp.hstack([t, x])
        for layer in self.layers[:-1]:
            x = jax.nn.elu(layer(x))
        return self.layers[-1](x)

    def step(self, t, x, dt):
        k1 = 0.5 * dt * self(t, x)
        x += dt * self(t + 0.5 * dt, x + k1)
        return x

    def sample(self, x, num_steps: int = 8):
        # midpoint method, second order.
        tt = jnp.linspace(0, 1, num_steps + 1)
        dt = 1 / num_steps
        for t in tt[:-1]:
            x = self.step(t, x, dt)
        return x
    
    def derivatives(self, t, x0, x1):
        xt = (1 - t) * x0 + t * x1
        v, v_t = jax.jvp(lambda t: self(t, xt), (t,), (jnp.ones_like(t,),))
        _, vv_x = jax.jvp(lambda x: self(t, x), (xt,), (v,))
        return v, v_t + vv_x


    def _loss_pinn(self, t, x0, x1):
        x_t = (1 - t) * x0 + t * x1
        v, v_t = jax.jvp(lambda t: self(t, x_t), (t,), (jnp.ones_like(t,),))
        _, vv_x = jax.jvp(lambda x_t: self(t, x_t), (x_t,), (v,))
        flow_matching = v - (x1 - x0) # rectified flow
        pde = (v_t + vv_x)
        return flow_matching, pde

    def _loss(self, t, x_0, x_1):
        x_t = (1 - t) * x_0 + t * x_1
        v = self(t, x_t)
        return v - (x_1 - x_0)
    

def loss_pinn(vf, xx_1, key):
    key1, key2 = jr.split(key)
    tt = jr.uniform(key1, shape=(xx_1.shape[0],))
    xx_0 = jr.normal(key2, xx_1.shape)
    flow_matching, pde = jax.vmap(vf._loss_pinn)(tt, xx_0, xx_1)
    loss_matching = (flow_matching**2).mean()
    loss_pde = (pde**2).mean()
    return loss_matching + 1e-2 * loss_pde

def pinn_guided_loss(vf, xx1, key):
    key1, key2 = jr.split(key)
    tt = jr.uniform(key1, (xx1.shape[0],))
    xx0 = jr.normal(key2, xx1.shape)
    _, pde = jax.vmap(vf._loss_pinn, (None, 0, 0))(0.5, xx0, xx1)
    flow_matching = jax.vmap(vf._loss)(tt, xx0, xx1)
    return (flow_matching**2).mean() + 1e-2 * (pde**2).mean()

def loss(vf, xx_1, key):
    key1, key2 = jr.split(key)
    tt = jr.uniform(key1, shape=(xx_1.shape[0],))
    xx_0 = jr.normal(key2, xx_1.shape)
    flow_matching = jax.vmap(vf._loss)(tt, xx_0, xx_1)
    return (flow_matching**2).mean()

def train(opt: optax.GradientTransformation, xx_1, epoch:int, mode: str):
    vf = Flow(dim=1)
    if "pinn" in mode:
        opt = jaxopt.OptaxSolver(loss_pinn, opt=opt)
    elif "vanilla" in mode:
        opt = jaxopt.OptaxSolver(loss, opt=opt)
    elif "guide" in mode:
        opt = jaxopt.OptaxSolver(pinn_guided_loss, opt=opt)
    
    key, subkey = jr.split(jr.key(0))
    state = opt.init_state(vf, xx_1, subkey)
    
    @jax.jit
    def step(vf, state, key):
        key, subkey = jr.split(key)
        vf, state = opt.update(vf, state, xx_1, subkey)
        return vf, state, key

    for it in (pbar:=trange(epoch)):
        vf, state, key = step(vf, state, key)
        pbar.set_postfix({"loss": f"{state.value:.3e}"})
    print("Done!")
    eqx.tree_serialise_leaves(f"checkpoints/{mode}.eqx", vf)