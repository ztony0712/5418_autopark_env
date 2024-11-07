# Neural Network design for Reinforcement Learning

This Jupyter Notebook implements and compares the performance of DDPG (Deep Deterministic Policy Gradient) and a simplified DDPG-Net. DDPG follows an Actor-Critic structure, while DDPG-Net relies on imitation learning without a Critic, providing a simpler alternative for continuous control tasks in reinforcement learning.


## Setup

To run this notebook, make sure you have all necessary dependencies installed. If you are using conda, create the environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment with:

```bash
conda activate ddpg_net_environment
```

Then launch Jupyter Notebook within this environment:

```bash
jupyter notebook
```

## Dependencies

This project relies on the following libraries, which are included in the `environment.yml` file:

- Python 3.12
- numpy 2.1.2
- pytorch 2.0.1
- matplotlib 3.8.0

## References

- [DDPG: Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)