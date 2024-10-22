# Offline Data Assimilation

## Correction Learning

Physics-based prediction models can be generally represented as:

$$
\frac{du}{dt} = F(u, t),
$$

where $u$ is a discretized vector.

To enhance the efficiency of Computational Fluid Dynamics (CFD), we introduce a learned correction method:

$$
\frac{du}{dt} = F(u, t) + f_\theta(u, t).
$$

In this context, $\theta$ is optimized to minimize discrepancies between the model's output and high-resolution observational data.

## Assimilation Learning

The primary goal of data assimilation is to derive high-quality initial conditions that are consistent with observations. 

The 3D-Var approach seeks to minimize the following objective:

$$
J(u) = \|u - u_b\|_{B^{-1}}^2 + \|y - H(u)\|_{R^{-1}}^2.
$$

Addressing this minimization problem in real time can be complex, prompting us to shift it to an offline setting. We can reformulate it as:

$$
\frac{du}{dt} = F(u, t) + f_\theta(u, y),
$$

where the loss function for the corresponding time step is defined as:

$$
L(\theta) = \|u_\theta - u\|_{B^{-1}}^2 + \|y - H(u_\theta)\|_{R^{-1}}^2.
$$

Upon minimizing this loss, the data assimilation process simplifies: we can compute the model solution using the learned assimilation term, where the correction corresponds to a network forward pass.

Importantly, evaluating $L(\theta)$ does not necessitate repeated runs of the predictive model. Using forward Euler time discretization, we obtain:

$$
u^{k+1} = u^k + \Delta t \left( F(u^k, k\Delta t) + f_\theta(u^k, y^{k+1}) \right)
$$

if $f_\theta$ is independent of $t$.

Given this, we can reformulate our objective function for the assimilation problem as:

$$
L_k(\theta) = \Delta t^2\| f_\theta(u^k, y^{k+1}) \|_{B^{-1}}^2 + \left \| y^{k+1} - H\left(u^{k+1} + \Delta t f_\theta(u^k, y^{k+1})\right) \right \|_{R^{-1}}^2,
$$

where $u^{k+1}$ denotes the numerical solution without correction.


## Thoughts

- I believe that many researchers are already utilizing deep learning models to train prediction models, followed by the application of methods such as EnKF, 3D-Var, and 4D-Var.
Given the availability of prediction models like FCN, PanguWeather, and NeuralGCM, it appears to be a promising approach to integrate data assimilation and evaluate performance.

- However, as a latecomer in this field, I find it challenging to catch up and produce a paper at this time.
I am contemplating a slightly different approach: maintaining the prediction model (Forward Euler) as fixed while employing deep learning techniques for data assimilation.