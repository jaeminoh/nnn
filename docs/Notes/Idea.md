# Machine learning + 3d variational data assimilation

## Correction Learning

A general form of physics-based prediction models can be expressed as:

$$
\frac{du}{dt} = F(u, t),
$$

where $u$ is a discretized vector.

To accelerate Computational Fluid Dynamics (CFD), we introduce a learned correction method:

$$
\frac{du}{dt} = F(u, t) + f_\theta(u, t).
$$

Here, $\theta$ is selected to minimize discrepancies between the model solution and high-resolution data.

## Assimilation Learning

The primary objective of data assimilation is to obtain high-quality initial conditions that align with observations. 

The 3D-Var approach aims to minimize:

$$
J(u) = \|u - u_b\|_{B^{-1}}^2 + \|y - H(u)\|_{R^{-1}}^2.
$$

Solving this minimization problem in real time can be challenging. Therefore, we propose shifting it to an offline task. We can reformulate it as:

$$
\frac{du}{dt} = F(u, t) + f_\theta(u, y, t),
$$

with a loss component for the same time point given by:

$$
L(\theta) = \|u_\theta - u\|_{B^{-1}}^2 + \|y - H(u_\theta)\|_{R^{-1}}^2.
$$

Once minimized, the data assimilation step becomes straightforward: we solve the model using the learned assimilation term, where the correction term corresponds to a network forward pass.

Evaluation of $L(\theta)$ does not require repeated predictive model run. Applying forward Euler time discretization yields:

$$
u^{k+1} = u^k + \Delta t \left( F(u^k, k\Delta t) + f_\theta(y^{k+1}) \right)
$$

if $f_\theta$ is independent of $t$.

Based on this observation, we can rewrite our objective function for the assimilation problem as:

$$
L(\theta) = \frac{1}{\Delta t^2}\| f_\theta(y^{k+1}) \|_{B^{-1}}^2 + \left \| y^{k+1} - H(u^{k+1} + \Delta t f_\theta(u^k, y^{k+1})) \right \|_{R^{-1}}^2,
$$

where $u^{k+1}$ represents the numerical solution without correction.

This is a way to parametrize solution.