# Lorenz 96

This example is from [Choi and Lee, 2024](https://arxiv.org/abs/2404.00154).

For $n = 1, \dots, N$, the following system of nonlinear ODEs describes the Lorenz 96 model:

$$
\frac{du_n}{dt} = (u_{n+1} - u_{n-2}) u_{n-1} - u_n + F,
$$

where $u_{-1} = u_{N-1}$, $u_{N+1} = u_1$, and $u_0 = u_N$.

Dimension: $N=128$;
Time domain: $[0, 200]$;
Forcing: $F=8$;
Initial condition: $(8.01, 8, \dots, 8)$.

Configuration for reference solution:
Time stepper: RK4;
Time step: $\Delta t = 0.01$.

Two figures present the results:
[[lorenz96_t5.pdf]] up to $t=5$, and [[lorenz96_t200.pdf]] up to $t=200$.

For generating the reference solution, consult with `lorenz96_ref.jl`.

## Data Assimilation
Observation time step $\Delta t_{obs} = 0.15$.
Observation contains 10% noise, following $N(0, \Gamma)$ with $\Gamma = (0.1s)^2I$, and $s$ is the standard deviation of the state solution.

Two terminologies:
- EnKF - **En**semble **K**alman **F**ilter
- ETKF - **E**nsemble **T**ransform **K**alman **F**ilter


