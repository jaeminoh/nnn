<h1 align='center'> NNN: Machine Learning-Based Nonlinear Nudging for Chaotic Dynamical Systems </h1>


## Abstract 
Nudging is an empirical data assimilation technique that incorporates an observation-driven control term into the model dynamics. The trajectory of the nudged system approaches the true system trajectory over time, even when the initial conditions differ. For linear state space models, such control terms can be derived under mild assumptions. However, designing effective nudging terms becomes significantly more challenging in the nonlinear setting. In this work, we propose neural network nudging, a data-driven method for learning nudging terms in nonlinear state space models. We establish a theoretical existence result based on the Kazantzis--Kravaris--Luenberger observer theory. The proposed approach is evaluated on three benchmark problems that exhibit chaotic behavior: the Lorenz 96 model, the Kuramoto--Sivashinsky equation, and the Kolmogorov flow.


## Setup
1. clone this repository: `git clone https://github.com/jaeminoh/nnn.git`
2. enter to the directory: `cd oda`
3. install [uv](https://docs.astral.sh/uv/getting-started/installation/).
4. setup via `bash setup.sh` and then activate the environment `source .venv/bin/activate`.

Or, install dependencies via `requirements.txt`.


## Examples
run `bash Lorenz96.sh`, `bash kursiv.sh`, or `bash KolmogorovFlow.sh`.


## Citation
If you find this repository useful in your research, please consider citing us!

```bibtex
@article{oh2025nnn,
  title={Machine Learning-Based Nonlinear Nudging for Chaotic Dynamical Systems},
  author={Oh, Jaemin and Lee, Jinsil and Hong, Youngjoon},
  journal={arXiv preprint arXiv:2508.05778},
  year={2025}
}
```