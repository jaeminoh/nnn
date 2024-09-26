# Kuramoto-Sivashinsky equation

This example is from [Choi and Lee, 2024](https://arxiv.org/abs/2404.00154).

The Kuramoto-Sivashinsky (K-S) equation is an evolutionary equation involving nonlinear convection, backward diffusion, and fourth-order diffusion as a model of instability of flames and phase turbulence in chemical reactions.

$$
u_t + uu_x + u_{xx} + u_{xxxx} = 0.
$$


$x\in [0, 32\pi)$, periodic domain.
Initial condition is $u_0(x) = \cos(x / 16) (1 + \sin(x / 16))$ which is a $32\pi$ periodic function.
$t\in [0, 10000]$.
Fourier collocation method, ETDRD4 time stepping with $\Delta t = 0.25$.

