# Kuramoto-Sivashinsky

- Model explanation: [[KuramotoSivashinsky]]
- Idea: [[OfflineDataAssimilation/index]]

## ToDo

- [ ] Input normalization


## Data Generation

For a numerical solver, see `julia_solver/kursiv.py`.

Solution profile looks like [[kursiv.pdf]].

- Reference Time Stepper: [ETDRK4](https://epubs.siam.org/doi/abs/10.1137/S1064827502410633)
- Spatial domain: $[0, 32\pi)$
- time domain: $[0, 10000]$
- Initial Condition: $u(x,0) = \cos(x/16) (1 + \sin(x/16))$
- 128 Fourier modes
- $\Delta t = 0.25$  
- Data Storage: The entire data is saved at `data/etdrk.npz`.


## Results

![](results/ensembles_lr0.001_epoch10000_noise10_rank128_test.png)