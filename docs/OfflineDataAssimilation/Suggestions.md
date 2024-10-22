# Idea and Suggestion

For method, please refer to [[OfflineDataAssimilation]].

## Inputs of the correction term

Originally, $f_\theta(u^k, y^{k+1})$.
However, optimization is difficult for this case.
Note that neural network initializations are designed to make gradients in $O(1)$ quantities.
The magnitude of $u$ and $y$ is about $8$, so appropriate normalization is necessary.