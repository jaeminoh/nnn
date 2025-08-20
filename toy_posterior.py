import jax
import numpy as np
from nnn.flow import Flow, train
import optax
import equinox as eqx
import matplotlib.pyplot as plt

np.random.seed(777)
xx_1 = np.random.randn(1024)  # samples
epoch = 10000
opt = optax.adamw(optax.cosine_decay_schedule(1e-3, epoch))
# vanilla
#train(opt, xx_1, epoch, "vanilla")
vector_field = Flow(dim=1)
vector_field = eqx.tree_deserialise_leaves("checkpoints/vanilla.eqx", vector_field)


# midpoint method
def midpoint_step(vector_field, t, x, dt):
    k1 = 0.5 * dt * vector_field(t, x)
    x += dt * vector_field(t + 0.5 * dt, x + k1)
    return x

# prior: x ~ N(0, 1)
def sample_from_prior(x, n_steps=8):
    tt = np.linspace(0, 1, n_steps+1)
    dt = tt[1] - tt[0]
    for t in tt:
        x = midpoint_step(vector_field, t, x, dt)
    return x

# observation ~ N(x, 1), np.ones(100)
def grad_log_likelihood(t, x, yy = np.ones(100)):
    return (yy - x).sum() * (1-t)
# posterior: x ~ N(100 / 101, 1/(101))
def sample_from_posterior(x, n_steps = 50):
    tt = np.linspace(1e-2, 1, n_steps+1)
    dt = tt[1] - tt[0]
    for t in tt:
        x = midpoint_step(lambda t, x: vector_field(t, x) + grad_log_likelihood(t, x), t, x, dt)
    return x

xx_0 = np.random.randn(1024)
xx_prior = jax.vmap(sample_from_prior)(xx_0)
xx_posterior = jax.vmap(sample_from_posterior)(xx_0)
fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.hist(xx_prior)
ax0.set_title(f"{np.mean(xx_prior):.2e}, {np.var(xx_prior):.2e}")
ax1.hist(xx_0)
ax1.set_title(f"{np.mean(xx_0):.2e}, {np.var(xx_0):.2e}")
plt.savefig("figures/prior_flow.pdf")

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.hist(xx_posterior)
ax0.set_title(f"{np.mean(xx_posterior):.2e}, {np.var(xx_posterior):.2e}")
xx_posterior_true = np.random.randn(1024) / np.sqrt(101) + 100 / 101  
ax1.hist(xx_posterior_true)
ax1.set_title(f"{np.mean(xx_posterior_true):.2e}, {np.var(xx_posterior_true):.2e}")
plt.savefig("figures/posterior_flow.pdf")
