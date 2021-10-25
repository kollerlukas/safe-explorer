# Implementation "Safe Exploration in Continuous Action Spaces"

## Introduction

This repository contains Pytorch implementation of paper ["Safe Exploration in Continuous Action Spaces" [Dalal et al.]](https://arxiv.org/pdf/1801.08757.pdf) [1]. 

The implemenation for the *Deep Determinitic Policy Gradient* (DDPG) [2] algorthim is taken from [here](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).

The Ball-Domain from [1] is implemented in a custom [OpenAI gym](https://gym.openai.com/) environment.

## Setup

The code requires Python 3.6+ and is tested with torch 1.1.0. To install dependencies run the following command.
```sh
pip install -r requirements.txt
```

## Training

A list of parameters and their default values is printed with the following command.
```sh
python -m safe_explorer.main --help
```

### BallND

The agent is trained by running the following command.
```sh
python -m safe_explorer.main --main_trainer_task ballnd
```

The training can be monitored via Tensorboard with the following command.
```sh
tensorboard --logdir=runs
```

## Results

To be updated.

## References
[1] Dalal, Gal, Krishnamurthy Dvijotham, Matej Vecerik, Todd Hester, Cosmin Paduraru, and Yuval Tassa (2018). “Safe Exploration in Continuous Action Spaces”. In: CoRR abs/1801.08757. arXiv: 1801.08757. url: http: //arxiv.org/abs/1801.08757.

[2] Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra (2016). “Continuous control with deep reinforcement learning”. In: CoRR abs/1509.02971.
