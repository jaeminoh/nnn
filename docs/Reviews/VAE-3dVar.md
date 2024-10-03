# Review: 3D-Var Data Assimilation Using a Variational Autoencoder

- [Peer-reviewed](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4708?af=R)
- [arXiv](https://arxiv.org/abs/2308.16073)

For a brief overview of data assimilation, please refer to [[DataAssimilation]].

## Motivation

3D-Var refers to three-dimensional variational data assimilation.

The 3D-Var method estimates the state by solving the following optimization problem, which combines the background state $ x_b $ and observations $ y $:

$$
\arg \min_x J(x) = J_b(x) + J_o(x),
$$

where

$$
J_b(x) = (x - x_b)^T B^{-1} (x - x_b),
$$

and

$$
J_o(x) = (y - H(x))^T R^{-1} (y - H(x)).
$$

In this formulation, $ B $ and $ R $ represent the covariance matrices for background and observation errors, respectively. The optimization yields a maximum-a-posteriori estimate, indicating that the optimal state $ x^\star $ corresponds to the peak of a Gaussian distribution, assuming independent Gaussian errors.

If the observation operator $ H(x) $ is nonlinear, the optimization problem becomes nonlinear as well, necessitating an iterative approach. However, calculating $ x \mapsto B^{-1} x $ can be computationally intensive. Specifically, for $ x \in \mathbb{R}^N $, the cost is $ O(N^2) $ if $ B^{-1} $ is readily available, but increases to $ O(N^3) $ when solving a linear system. For instance, with ERA5 data at a resolution of $ 720 \times 1440 $, $ N $ can be as large as $ 10^6 $.

This paper introduces a Variational Autoencoder (VAE) for dimensionality reduction. The encoder maps the state $ x $ to a latent variable $ z \in \mathbb{R}^{100} $, while the decoder $ D $ transforms $ z $ back to $ x $. Consequently, the 3D-Var objective function is reformulated as:

$$
(z - z_b)^T B_z^{-1} (z - z_b) + (y - H(D(z)))^T R^{-1} (y - H(D(z))).
$$

If the latent-background-error covariance matrix $ B_z $ is known, the computational cost for the first term is significantly reduced from $ O(10^{12}) $ to $ O(10^4) $.