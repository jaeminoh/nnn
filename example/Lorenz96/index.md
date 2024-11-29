# Lorenz96

- Model explanation: [[Lorenz96]]
- Idea: [[OfflineDataAssimilation/index]]

## ToDo

- [ ] Input normalization


## Data Generation

**Reference Time Stepper:** [Tsit5](https://www.sciencedirect.com/science/article/pii/S0898122111004706)

**Initial Condition:**  
$$u_0 = [8.01, 8, 8, \ldots, 8]^T \in \mathbb{R}^{N_x}$$

**Ensemble Generation:**  
For $i = 1, \ldots, K$, apply 1% Gaussian noise to $u_0$:  
$$u_0^i = u_0 + 0.01 \epsilon_i$$  
where $\epsilon_i \sim N(0, 1)$.  
The ensemble $\{u^i\}$ is created by solving the Lorenz96 model using the respective initial conditions.

**Data Storage:**  
Ensemble data is saved in `data/Tsit.npz`.  
The array dimensions are $(K, N_t, N_x)$, where $K$ is the ensemble size, $N_t$ is the number of time steps, and $N_x$ is the vector length.