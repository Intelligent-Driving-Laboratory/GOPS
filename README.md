# GOPS (General Optimal control Problem Solver)

[![Read the Docs](https://img.shields.io/readthedocs/gops)](https://gops.readthedocs.io/en/latest/)

Copyright © 2022 Intelligent Driving Laboratory (iDLab). All rights reserved.

## Description
Optimal control is an important theoretical framework for sequential decision-making and control of industrial objects, especially for complex and high-dimensional problems with strong nonlinearity, high randomness, and multiple constraints.
Solving the optimal control input is the key to applying this theoretical framework to practical industrial problems.
Taking Model Predictive Control as an example, computation time solving its control input relies on receding horizon optimization, of which the real-time performance greatly restricts the application and promotion of this method.
In order to solve this problem, iDLab has developed a series of full state space optimal strategy solution algorithms and the set of application toolchain for industrial control based on Reinforcement Learning and Approximate Dynamic Programming theory.
The basic principle of this method takes an approximation function (such as neural network) as the policy carrier, and improves the online real-time performance of optimal control by offline solving and online application.
The GOPS toolchain will cover the following main links in the whole industrial control process, including control problem modeling, policy network training, offline simulation verification, controller code deployment, etc.
GOPS currently supports the following algorithms:
- [Deep Q Network (DQN)](https://arxiv.org/abs/1312.5602)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477)
- [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783)
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [Distributional Soft Actor-Critic (DSAC)](https://arxiv.org/abs/2001.02811)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Infinite-Horizon Approximate Dynamic Programming (INFADP)](https://link.springer.com/book/10.1007/978-981-19-7784-8)
- [Finite-Horizon Approximate Dynamic Programming (FHADP)](https://link.springer.com/book/10.1007/978-981-19-7784-8)
- [Mixed Actor-Critic (MAC)](https://ieeexplore.ieee.org/document/9268413)
- [Mixed Policy Gradient (MPG)](https://arxiv.org/abs/2102.11513)
- [Separated Proportional-Integral Lagrangian (SPIL)](https://arxiv.org/abs/2102.08539)

## Installation
GOPS requires:
1. Windows 7 or greater or Linux.
2. Python 3.6 or greater (GOPS V1.0 precompiled Simulink models use Python 3.8). We recommend using Python 3.8.
3. (Optional) Matlab/Simulink 2018a or greater.
4. The installation path must be in English.

You can install GOPS through the following steps:
```bash
# clone GOPS repository
git clone https://github.com/Intelligent-Driving-Laboratory/GOPS.git
cd GOPS
# create conda environment
conda env create -f gops_environment.yml
conda activate gops
# install GOPS
pip install -e .
```

## Documentation
The tutorials and API documentation are hosted on [gops.readthedocs.io](https://gops.readthedocs.io/en/latest/).

## Quick Start
This is an example of running finite-horizon Approximate Dynamic Programming (FHADP) on inverted double pendulum environment. 
Train the policy by running:
```bash
python example_train/fhadp/fhadp_mlp_idpendulum_serial.py
```
After training, test the policy by running:
```bash
python example_run/run_idp_fhadp.py
```
You can record a video by setting `save_render=True` in the test file. Here is a video of running a trained policy on the task:

<div align=center>
<video src=https://github.com/Intelligent-Driving-Laboratory/GOPS/assets/113587370/a095e8a2-5732-470e-9116-2d592eb8e3c6></video>
</div>

## WeChat Group
In order to make it easier for everyone to use GOPS and build a good community, we have established a WeChat group for GOPS users and invite interested users to join by scanning the QR code below.
Developers will answer questions for users in the group when using GOPS, and will fix problems in GOPS based on user feedback. In addition, the release of a new version of GOPS will also be notified in the group.

Thanks to all users for your support of GOPS and to all developers for your contributions to GOPS. Let's work together to make GOPS a valuable, easy-to-use, and popular software!

<div align=center>
<img src=https://github.com/Intelligent-Driving-Laboratory/GOPS_DOC/blob/master/docs/source/figures%26videos/QR_code.jpg width="40%">
</div>
