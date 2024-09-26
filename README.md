# Data Assimilation in the Latent Space

## How to execute?
`git clone https://github.com/jaeminoh/latent_da.git`

Install [Julia](https://julialang.org/downloads/)

Open Julia REPL (read-evaluate-print loop)

Type
```julia
] # package mode
activate . # activate current env
```


## Lorenz 96
This example is from [Choi and Lee, 2024](https://arxiv.org/abs/2404.00154).

$N=128$, $F=8$.

For $n = 1, \dots, N$,

$$
\frac{du_n}{dt} = (u_{n+1} - u_{n-2}) u_{n-1} - u_n + F,
$$
where $u_{-1} = u_{N-1}$, $u_{N+1} = u_1$, and $u_0 = u_N$.

Initial state is $(8.01, 8, \dots, 8)$.