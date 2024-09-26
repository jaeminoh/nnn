# Auto-Encoding Variational Bayes

[Kingma and Welling, 2013](https://arxiv.org/abs/1312.6114)

@misc{kingma2022auto,
      title={Auto-Encoding Variational Bayes}, 
      author={Diederik P Kingma and Max Welling},
      year={2022},
      eprint={1312.6114},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1312.6114}, 
}

For an iid random sample $x_i \sim p$,
we are often interested in the data distribution $p$.
For example, $x_i$ could be a high quality image.
Finding $p$ enables us to generate high quality images.
However, it is not an easy task.

We assume that there is a latent variable $z$ so that the actual data generating process is a two step:
$z \sim p(z)$ and then $x \sim p(x|z)$.
Unfortunately, this latent variable $z$ is unknown in general.
Identifying a latent variable is a problem of identifying "underlying dynamics," which is very difficult.
Auto-Encoding variational Bayes provide a framework to identify $z$, $p(x|z)$ (decoder) and $q(z|x)$ (encoder).