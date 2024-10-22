# Neural Controlled Differential Equations

Neural ordinary differential equations (NODEs) have a form of

$$
    \frac{dz}{dt} = f_\theta(z, t),
$$
which may be rewritten as

$$
    dz = f_\theta(z, t)dt.
$$

Neural CDE has a similar form:

$$
    dz = f_\theta(z)dx.
$$

Here, $z$ is a hidden state vector, and $x$ is an observation.

When a time series of $x = (x_{t_0}, \dots, x_{t_N})$  is given, NCDE first interpolates the series into a path $x: [t_0, t_N] \rightarrow \mathbb{R}^{d_x}$, and then solve $dz = f_\theta(z) dx$ by Riemann-Stiljes integral, or solve $dz = f_\theta(z) \frac{dx}{dt}dt$ if the path is differentiable in time.

Forward Euler discretization of NCDE results in recurrent neural networks. So NCDE may be viewed as a continuous generalization of RNNs. Due to continuity, NCDE can deal with irregular time series too.

Actually, our offline data assimilation format

$$
    x^{k+1} = x^k + \Delta t F(x^k) + \Delta f_\theta(x^k, y^{k+1})
$$

is a special form of NCDE. Consider the following equation:

$$
    dx = F(x)dt + f_\theta(x)dy
$$

a discretization derives

$$
    x^{k+1} - x^k = F(x^k)\Delta t + f_\theta(x^k) (y^{k+1} - y^k)
$$

or

$$
    x^{k+1} - x^k = F(x^k)\Delta t + f_\theta(x^k) \frac{dy^k}{dt}\Delta t.
$$

At first glance, $f_\theta(x^k, y^{k+1})$ seems more flexible, as it depends on $y^{k+1}$ in a nonlinear way.
