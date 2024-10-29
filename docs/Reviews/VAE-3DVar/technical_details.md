# VAE-3DVAR: Technical Details

Training Variational Autoencoders (VAEs) involves several important technical aspects, including data pre-processing and the selection of hyperparameters for neural networks.

- [VAE-3DVAR: Technical Details](#vae-3dvar-technical-details)
  - [Pre-processing](#pre-processing)
    - [Normalization](#normalization)
    - [Padding](#padding)
  - [Hyperparameters](#hyperparameters)
    - [Loss Function for Training VAE](#loss-function-for-training-vae)

## Pre-processing

Both training and test data must be properly processed before being input into the model. However, this process can be time-consuming. To streamline this, we can convert the temperature field into spectral coefficient space (see [[SphericalHarmonics]]) and use this representation as input for the model, replacing the [Padding](#padding) phase.


### Normalization
Normalization is essential, particularly for the temperature field input, which is measured in Kelvin (around 290 K). Without proper normalization, the optimization process can become unstable, leading to poor reconstruction quality. For example:
- True: [[x_era5.pdf]]
- Reconstructed: [[x_recon.pdf]]

To standardize the temperature fields, we subtract the mean and divide by the standard deviation, calculated on a day-of-year basis. Let $d$ represent the day of the year. The standardization procedure is given by:

$$
x_d^s = \frac{x_d - \mu_d}{\sigma_d}
$$

where

$$
\mu_d = \frac{1}{Y}\sum_{y=1}^Y x_{d, y}, \quad \sigma_d = \sqrt{\frac{1}{Y-1}\sum_{y=1}^Y (x_{d,y} - \mu_d)^2 }.
$$

Once day-of-year statistics are computed, the destandardization process is straightforward.

**Considerations:**

- [ ] Clarify whether to use $Y$ or $Y-1$ in the denominator.
- [ ] Address handling of February 29th.


### Padding
To accurately represent the Earth's geometry, cyclic padding is applied in the longitudinal direction. This may be replaced with spherical harmonics.


## Hyperparameters

### Loss Function for Training VAE
The VAE loss comprises two components: reconstruction loss and regularization loss:

$$
\mathcal{L}_\mathrm{VAE} = \mathcal{L}_\mathrm{rec} + \mathcal{L}_\mathrm{reg},
$$

where the reconstruction loss quantifies the difference between the original and reconstructed fields, while the regularization loss ensures that $p(z)$ aligns with a chosen target distribution (e.g., standard normal). In this work, Huber loss is utilized as the reconstruction loss:

$$
\mathcal{L}_\mathrm{rec} (\hat{x}, x) = \frac{A}{n}\sum_{i=1}^n L_\delta (\hat{x}_i, x_i),
$$

and

$$
\mathcal{L}_\mathrm{reg} = -\log p(z) + \log q_\phi (z | x).
$$

Here, $A$ is a hyperparameter that influences the balance between reconstruction quality and the adherence of $z$ to a standard normal distribution; a larger $A$ emphasizes reconstruction accuracy, while a smaller $A$ prioritizes the regularization term.