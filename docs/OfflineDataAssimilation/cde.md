# NCDE + ODA
The primary goal is to produce reanalysis immediately after an observation is obtained.

**Notation**

- $x$: state
- $y$: observation
- Subscript $f$: forecasting
- Subscript $a$: analysis
- Superscript: time indexing
- $M$: forecasting model
- $H$: observation operator

## Review: [[OfflineDataAssimilation/index]]
Forecasting is

$$
x_f^{k+1} = x_a^k + \Delta t F (x_a^k),
$$

and analysis step is

$$
x_a^{k+1} = x_f^{k+1} + \Delta t f_\theta(x_f^{k+1}, y^{k+1}).
$$

For convenience, we introduce $M$ and $N$ such that $x_f^{k+1} = M(x_a^k)$ and $x_a^{k+1} = N(x_f^{k+1})$. Note that $N$ is depending on network parameters $\theta$.
The network parameters $\theta$ are determined by minimizing:

$$
L(\theta) = \sum_{k=1}^K \| (N \circ M)^k(x_a^0) - M \circ (N \circ M)^{k-1}(x_a^0) \|_B^2 + \| y^k - H \circ (N\circ M)^k(x_a^0) \|_R^2.
$$

However, this approach has not yielded satisfactory results. Why?

- If the first term is small, then $f_\theta(x_f^{k+1}, y^{k+1})$ must also be small. Therefore, the first term could be replaced with weight decay (e.g., AdamW[@loshchilov2017fixing], Lion[@chen2023symbolic]).

- The second term (data fit loss) is motivated by the 3D-Var cost function, but despite including future observations, it fails to exploit the 4D structure of the data.


## Method

Given observation data $y^0, \ldots, y^K$, we can consider the following [[NeuralCDE]]:

$$
dx = F(x) dt + f_\theta(x) dy.
$$

If $y$ is a [brownian motion](https://en.wikipedia.org/wiki/Brownian_motion), this formulation effectively becomes a neural stochastic differential equation. The term $F(x) dt$ indicates that our NCDE resembles the Lorenz 96 model with an additional correction control term.

However, preliminary experiments indicate that the generalization performance is poor and the model is unstable in the presence of noise. (Why?)

A discretization leads to:

$$
x^{k+1} - x^k = F(x^k) \Delta t + f_\theta(x^k) (y^{k+1} - y^k),
$$

or equivalently,

$$
x^{k+1} - x^k = F(x^k) \Delta t + f_\theta(x^k) \frac{dy^k}{dt} \Delta t.
$$


## Comparison vs [Assimilation Learning](index.md#assimilation-learning)

At first glance, $f_\theta(x_f^{k+1}, y^{k+1})$ appears more flexible since it nonlinearly depends on $y^{k+1}$. But in fact, the form of controlled differential equations admits rich expressibility. For more detail, see Appendix C[@kidger2020neural].

### Case study on [[Lorenz96]]

- Reference Solution: The underlying dynamics of weather are assumed to be obtained via high-order method. However, since there is always some error in observations, we add white noise to account for this variability. Reference solution is generated with Tsit5[@tsitouras2011runge]. Observation is reference solution + Gaussian noise.

- Forward Euler Solution: Weather predictions are made using the Forward Euler method. Due to the chaotic behavior of the Lorenz96 model, truncation errors accumulate rapidly, causing the predictions to diverge quickly from the reference solution, which reflects the actual weather dynamics. Prediction model is Lorenz96 + Forward Euler solver. Due to local truncation error, prediction largely deviates from the reference solution.

- Observational Data Integration: Data assimilation adjusts the Forward Euler solution (predictions) based on observed values ($y$). This process ensures that the predictions remain consistent with the reference solution, which represents the actual dynamics of the weather. Loss function is MSE(prediction, observation).

**Result**

When there is no noise, both achieved $O(10^{-3})$ relative $L^\infty$ errors.

NCDE:
![NCDE](figures/ncde_noise0.png)

Base:
![BASE](figures/base_noise0.png)

However, for noisy case, Assimilation learning achieved far accurate result. For 1% noise case, base method achieved 3.26e-3, whereas ncde method achieved 2.42e-1, worse than pure forward euler (2.08e-2). See the table below.

NCDE:
![NCDE](figures/ncde_noise1.png)

Base:
![BASE](figures/base_noise1.png)


Summary Table:

|   | 0%  | 1%  | 5%  | Forward Euler  |
|---|---|---|---|---|
| BASE  |  3.24e-3 | 3.26e-3  | 6.37e-3  | 2.08e-2  |
| NCDE  | 2.00e-3  | 2.42e-1  | 1.16e+0  |   |

In short, NCDE approach is sensitive to noise?

- Path construction procedure is hermite spline interpolation. Noisy observation might decrease regularity, resulting in a poor approximation of the true path.
- Construction of a path with least square would be a remedy.
- Or, random chebyshev interpolation[@matsuda2024polynomial] is also possible (?) to construct a path.