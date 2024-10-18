import numpy as np
from tqdm import trange


def solve(solver_step, net, state, u0, yy, maxiter=5000):
    min_loss = np.inf
    for it in (pbar := trange(1, 1 + maxiter)):
        net, state = solver_step(net, state, u0, yy)
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{(loss:= state.value):.3e}"})
            if np.isnan(loss):
                break
            elif loss < min_loss:
                min_loss = loss
                opt_net = net

    print(f"Done! min_loss: {min_loss:.3e}, final_loss: {state.value:.3e}")
    return opt_net, state


def load_data(name: str, unroll_length: int, noise_level: int, seed: int = 0):
    np.random.seed(seed)
    d = np.load(f"data/{name}.npz")
    tt = d["tt"]
    u0 = d["sol"][:, 0]
    uu_ref = d["sol"][:, 1 : 1 + unroll_length]
    yy = uu_ref + 0.01 * noise_level * np.random.randn(*uu_ref.shape)
    return tt, u0, uu_ref, yy
