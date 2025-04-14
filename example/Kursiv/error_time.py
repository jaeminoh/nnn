import matplotlib.pyplot as plt
from oda.filters import ClassicFilter as Filter
from oda.models import Kursiv
from oda.data_utils import DataLoader
import equinox as eqx
from oda.networks import SimpleCorrector as Net
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

net = Net(
    kernel_size=5,
    stride=4,
    num_spatial_dim=1,
    hidden_channels=128,
)
fname = "kursiv_lr0.001_epoch1000_noise0_rank128"
net = eqx.tree_deserialise_leaves(f"data/{fname}.eqx", net)
model = Kursiv(sensor_every=4, d_in=1)
filter = Filter(model=model, observe=model.observe)
data_loader = DataLoader(model.observe, noise_level=1)

tt, u0, uu_ref, yy = data_loader.load_test("data/test.npz", 25000)
uu_base = filter.model.solve(u0, tt)
uu_f, uu_a = filter.unroll(net, u0 + np.random.standard_normal(u0.size) * 0.5, yy)

L2_t = jax.vmap(jnp.linalg.norm)(uu_ref - uu_f)
plt.semilogy(tt[1:], L2_t)
plt.tight_layout()
plt.savefig("data/kursiv_l2error_time.pdf")
