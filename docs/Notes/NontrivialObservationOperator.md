# Dealing with nontrivial observation operators

The simplest nontrivial observation operator would be

$$
H: \begin{bmatrix}
O \\
O \\
O \\
O
\end{bmatrix}
\mapsto 
\begin{bmatrix}
O \\
X \\
O \\
X 
\end{bmatrix}.
$$

$H, H^2, H\otimes H$ are simple, yet our method does not work well for them.
That might be a reason why there is no (self-supervised loss + without ensemble)  method.

To deal with them, we have to use a more sophisticated inductive bias (network architecture).
Recalling the first improvement comes from putting the innovation $y - H(u^f)$ into neural network, we can think of the following modifications for the input of the neural network:

- $y - H_\theta(u^f)$, learnable observation operator,
- $H_\theta^*(y) - u^f$, learnable inverse (transpose) observation operator.

The self-supervised loss motivated by the 4DVAR method can be modified accordingly:

- $\sum_{k}\left( y - H_\theta(\tilde{u}_k^f) \right)^2$,
- $\sum_{k}\left( H_\theta^*(y) - \tilde{u}_k^f \right)^2$. (tilde notation means network outputs.)

Or, we may use the original self-supervised loss $\sum_{k}\left( y - H(\tilde{u}_k^f) \right)^2$.

However, these modifications resulted in no improvement in our experiments 🥲.

## Why?

- Even though $H$ removes some information at specific locations, the Kalman filter updates forecast to analysis for removed locations based on error covariance matrix. However, our method has only considered the identity matrix for the error covariance matrix.

- 