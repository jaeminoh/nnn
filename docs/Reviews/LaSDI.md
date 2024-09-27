# Latent space dynamics identification

Computational simulation based on differential equations is often expensive, especially when the physical dimension is high.
Reduced order modelling (ROM) is a technique to reduce the computational cost trying to maintain the accuracy as much as possible.

Latent space dynamics identification (LaSDI) is a ROM framework for dynamical systems.
It consists of compress the original vector in physical space, propagate in the latent space, and then decompress the latent output to the physical space.
It is conceptually the same with transform an initial condition to frequency space and then propagates in time, and then revert back to the physical space (spectral methods).
However, in LaSDI framework, propagation rule is unknown.




@article{fries2022lasdi,
  title={Lasdi: Parametric latent space dynamics identification},
  author={Fries, William D and He, Xiaolong and Choi, Youngsoo},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={399},
  pages={115436},
  year={2022},
  publisher={Elsevier}
}

@article{park2024tlasdi,
  title={tLaSDI: Thermodynamics-informed latent space dynamics identification},
  author={Park, Jun Sur Richard and Cheung, Siu Wun and Choi, Youngsoo and Shin, Yeonjong},
  journal={arXiv preprint arXiv:2403.05848},
  year={2024}
}
