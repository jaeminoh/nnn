# Neural Controlled Differential Equations

Neural Ordinary Differential Equations (NODEs) take the form:

$$
\frac{dz}{dt} = f_\theta(z, t),
$$

which can be rewritten as:

$$
dz = f_\theta(z, t) dt.
$$

Neural Controlled Differential Equations (NCDEs) have a similar structure:

$$
dz = f_\theta(z) dx.
$$

Here, $z$ represents a hidden state vector, and $x$ is an observation.

When given a time series $x = (x_{t_0}, \ldots, x_{t_N})$, the NCDE first interpolates the series into a path $x: [t_0, t_N] \rightarrow \mathbb{R}^{d_x}$ and then solves $dz = f_\theta(z) dx$ using a Riemann-Stieltjes integral. If the path is differentiable in time, it can also solve:

$$
dz = f_\theta(z) \frac{dx}{dt} dt.
$$

The Forward Euler discretization of NCDE yields recurrent neural networks, allowing NCDEs to be seen as a continuous generalization of RNNs. This continuity enables NCDEs to handle irregular time series effectively.