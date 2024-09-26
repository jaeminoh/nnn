# Lorenz96

This example is from [Choi and Lee, 2024](https://arxiv.org/abs/2404.00154).

@misc{choi2024sampling,
      title={Sampling error mitigation through spectrum smoothing in ensemble data assimilation}, 
      author={Bosu Choi and Yoonsang Lee},
      year={2024},
      eprint={2404.00154},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2404.00154}, 
}

$N=128$, $F=8$.

For $n = 1, \dots, N$,

$$
\frac{du\_n}{dt} = (u\_{n+1} - u\_{n-2}) u\_{n-1} - u\_n + F,
$$
where $u_{-1} = u_{N-1}$, $u_{N+1} = u_1$, and $u_0 = u_N$.

Initial state is $(8.01, 8, \dots, 8)$.

The result is presented here: [[lorenz96.pdf]]