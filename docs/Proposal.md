# Data Assimilation in Latent Spaces

For concrete understanding, let us assume that we'd like to combine the primitive equation and radar observation data to predict the weather accurately.

Assume that we had introduced a grid of $256 \times 128 \times 16 = 2^{19}$.
Discretization of the primitive equation according to this grid results in a system of ODE with approximately $5 \cdot 10^5$ variables.

Let $X(t)$ be the discretized solution at time $t$.
Then the generic form of the discretized primitive equation looks like

$$
\frac{dX(t)}{dt} = f(X(t), t).
$$

Let $Y$ be the observation (such as radar reflectivity).
The observation process can be represented as

$$
Y = H(X) + \epsilon,
$$

where $\epsilon$ is a Gaussian noise.

Then the Kalman filter estimates the corrected solution by

$$
\tilde{X}_k = X_k + W_k(Y_k - H(X_k)).
$$

However, $W_k$ is often too large to compute.
3D-Var or 4d-Var uses $W_k = W$, so it is relatively cheap.
But time-dependent weight matrices $W_k$ are expensive.
Why? It involves $B_k^{-1}$, the inverse covariance matrix of a prior distribution.
It contains $0.5 * (5 \cdot 10^5)^2$ entries.
If we are in single precision arithmetic, the storage requirement is more than $4 \cdot 10^{11}$ Bytes $= 4 \cdot 10^{11} / 2^{30}$ GigaBytes $\approx 400$ GigaBytes.

In summary, we have to play with a 400GB matrix **at each time step**.
If we reduce the resolution, then the accuracy will decrease.