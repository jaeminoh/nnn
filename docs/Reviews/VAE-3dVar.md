# Review: 3D-Var Data Assimilation Using a Variational Autoencoder
For a brief overview of data assimilation, please refer to [[DataAssimilation]].

- [Peer-reviewed](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4708?af=R)
- [arXiv](https://arxiv.org/abs/2308.16073)




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

In this formulation, $ B $ and $ R $ represent the covariance matrices for background and observation errors, respectively. The optimization yields a maximum-a-posteriori estimate, indicating that the optimal state $ x^\star $ corresponds to the peak of a posterior distribution, assuming independent Gaussians.

If the observation operator $ H(x) $ is nonlinear, the optimization problem becomes nonlinear as well, necessitating an iterative approach. However, calculating $ x \mapsto B^{-1} x $ can be computationally intensive. Specifically, for $ x \in \mathbb{R}^N $, the cost is $ O(N^2) $ if $ B^{-1} $ is readily available, but increases to $ O(N^3) $ when solving a linear system. For instance, with ERA5 data at a resolution of $ 720 \times 1440 $, $ N $ can be as large as $ 10^6 $.

This paper introduces a Variational Autoencoder (VAE) for dimensionality reduction. The encoder maps the state $ x $ to a latent variable $ z \in \mathbb{R}^{100} $, while the decoder $ D $ transforms $ z $ back to $ x $. Consequently, the 3D-Var objective function is reformulated as:

$$
(z - z_b)^T B_z^{-1} (z - z_b) + (y - H(D(z)))^T R^{-1} (y - H(D(z))).
$$

If the latent-background-error covariance matrix $ B_z $ is known, the computational cost for the first term is significantly reduced from $ O(10^{12}) $ to $ O(10^4) $.


## Derivation of the 3D-Var
We start with an estimate of the current state, denoted as $ x_b $. Based on this information, we assume that the true state $ x $ is close to $ x_b $, which can be expressed as:

$$
x \sim N(x_b, B),
$$

where $ B $ is a known covariance matrix. This normal distribution assumption reflects our belief about the uncertainty in the state estimate.

Next, we consider an observation $ y $ related to the true state $ x $. The observation process can be modeled as:

$$
y = H(x) + \epsilon,
$$

where $ \epsilon $ represents the measurement error. We assume that the measurement error follows a normal distribution:

$$
y | x \sim N(H(x), R),
$$

with $ R $ as the covariance matrix for the observation noise.

Since we have partially observed the true state $ x $ through $ y $, we need to update our belief about $ x $ using Bayes' theorem:

$$
p(x | y) = \frac{p(y | x) p(x)}{\int p(y | x) p(x) \, dx}.
$$

Given that we can express both $ p(x) $ and $ p(y | x) $ analytically, determining $ p(x | y) $ becomes manageable. To simplify the computation, we can ignore constants when focusing on maximizing the posterior density.

The integral in the denominator corresponds to $ p(y) $ and can therefore be disregarded. For the other constants, we have:

$$
p(x) = |2\pi B|^{-d_x / 2} \exp\left(-\frac{1}{2}(x - x_b)^T B^{-1} (x - x_b)\right),
$$

and

$$
p(y | x) = |2\pi R|^{-d_y / 2} \exp\left(-\frac{1}{2} (y - H(x))^T R^{-1} (y - H(x))\right).
$$

Since the leading scaling constants are not necessary for our maximization, we find that the posterior density is proportional to:

$$
p(x | y) \propto \exp\left(-\frac{1}{2} \left((x - x_b)^T B^{-1} (x - x_b) + (y - H(x))^T R^{-1} (y - H(x))\right)\right).
$$

The maximum a posteriori (MAP) estimator is then the value of $ x $ that maximizes this posterior density. Thus, we have derived the basis for the 3D variational data assimilation method.


## Proposals

Here we provide some proposals.

### Parametrize $x$ with neural networks.
Assume that $\mathrm{dim}(y) \ll \mathrm{dim}(x)$. In this scenario, the main challenge in optimization is computing $(x - x_b)^T B^{-1} (x - x_b)$. By parametrizing $x_b$ and $x$ with $\theta_b$ and $\theta$, respectively, we can reduce the number of unknowns and leverage the ease of training neural networks. However, we have to model the error covariance matrix $B_\theta$: to minimize $J(\theta) = (\theta - \theta_b)^T B_\theta^{-1} (\theta - \theta_b) + (y - H(x_\theta))^T R^{-1} (y - H(x_\theta))$.

### Add another penalty.
Certain physical quantities, such as mass, momentum, and energy, must be conserved. However, an assimilated state $\hat{x}$ may not necessarily adhere to these conservation laws. It’s important to note that $x_b$ satisfies a specific set of conservation laws, as it is derived from solving the primitive equations. By introducing a penalty term like $\int x d\mu - \int x_b d\mu$, we may enhance the quality of the reanalysis.