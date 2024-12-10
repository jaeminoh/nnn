# Review: Learning Semilinear Neural Operators: A Unified Recursive Framework for Prediction and Data Assimilation

Overall, the paper is well written and easy to read. However, the experiments section is not very easy to follow. For instance, Table 1 and Table 2 do not have appropriate explanations in the main text.

To summarize their method, we have to begin with a generic form of semilinear PDEs.

$$
u_t = Lu + N(u, t),
$$

where $L$ is a linear operator and $N$ is a nonlinear operator. When the model is incorrect and there is an observation process, then we may consider the observer design,

$$
u_t = Lu + N(u, t) + K(t)[y - Cu],
$$

similar to continuous time Kalman gain. The linearity of $L$ induces a semigroup operator $T: T(t+t') = T(t)T(t')$. Then, the solution to the above equation can be written as follows.

$$
u(t) = T(t)u(0) + \int_0^t T(t-s)N(u(s), s)ds + \int_0^t T(t-s)K(s)[y - Cu(s)]ds.
$$

The first term, 

$$T(t)u(0) + \int_0^t T(t-s)N(u(s), s)ds,$$

is called *prediction*, and the second term,

$$\int_0^t T(t-s)K(s)[y - Cu(s)]ds,$$
is called *correction*.

It is written that the authors' goal was to exploit the structure of semilinear PDEs to design an effective neural operator architecture. However, there was no such structure-exploitation. They only exploited this prediction-correction structure, by

$$
u_k^f = u_{k-1}^a + \mathrm{FNO}_\theta(u_{k-1}^a),
$$

and

$$
u_k^a = u_k^f + K_\theta(u_k^f)[y_k - E_\theta(u_k^f)].
$$

Especially, experimental results are not very convincing (Table 2). FNO and NODA should achieve similar performance for $\alpha=0$ case, because

- for FNO, accurate $u(t_{H-1})$ (the last sentence of the page 6.)
- for NODA, accurate up to $u(t_H)$ due to the warm-up phase.
- the difference between FNO and NODA is only residual formulation.

However, the accuracy deviates a lot. This might come from three reasons; first, simulation step is too large, making $\hat{u}(t_H)$ inaccurate for FNO; second, residual formulation is very effective; third, not sufficient hyperparameter tuning for FNO.