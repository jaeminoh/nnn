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


def load_data(unroll_length: int, noise_level: int, seed: int = 0):
    np.random.seed(seed)
    d = np.load("data/Tsit.npz")
    tt = d["tt"]
    Nt, Nx = d["sol"].shape

    assert (Nt - 1) % unroll_length == 0, "unroll_length should divide (Nt-1)!"
    N_traj = (Nt - 1) // unroll_length

    u0 = np.zeros((N_traj, Nx))
    uu_ref = np.zeros((N_traj, unroll_length, Nx))

    for i in range(N_traj):
        _start = unroll_length * i
        u0[i] = d["sol"][_start]
        uu_ref[i] = d["sol"][_start + 1 : _start + unroll_length + 1]

    assert np.allclose(u0[1:], uu_ref[:-1, -1]), "index error!"

    yy = uu_ref + 0.01 * noise_level * np.random.randn(*uu_ref.shape)
    return tt, u0, uu_ref, yy


def normalize(*arrays):
    return [(array - 8) / 8 for array in arrays]


def denormalize(*arrays):
    return [(array * 8 + 8) for array in arrays]
