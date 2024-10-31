# Sketches

Kalman filter is a very efficient data assimilation technique.
However, for nonlinear prediction models, update formulae should be modified based on linearization (extended Kalman filter).
This procedure could involve computing model adjoints, often expensive.
Linear model is often desirable, because we may exploit well-established numerically accurate and stable linear solvers for many cases. For example, consider a dyanmical system (autonomous, generic form):

$$
\frac{du}{dt} = F(u).
$$

If $F$ is linearly depending on $u$, i.e. $F(u) = Fu$, then we say this dynamical system is linear. However, if $F$ is nonlinear, then Kalman filter requires the following linearized form:

$$
\frac{du}{dt} = \frac{dF}{du} u.
$$

However, it is also an approximation. Note that high-order terms are ignored.

In this scenario, can we convert nonlinear prediction model into linear prediction model?
There are some works.

[[KoopmanOperator]]

- [ ] [SynDI](https://www.pnas.org/doi/abs/10.1073/pnas.1517384113)
- [ ] [Cole-Hopf transform](https://en.wikipedia.org/wiki/Cole–Hopf_transformation)