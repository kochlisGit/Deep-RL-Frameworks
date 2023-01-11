# DRL-Frameworks
Comparison of different Deep Reinforcement Learning (DRL) Frameworks. This repository currently supports TF-agents, RLlib examples

# TF-Agents
TF-Agents makes designing, implementing and testing new RL algorithms easier, by providing well tested modular components that can be modified and extended. It enables fast code iteration, with good test integration and benchmarking. While not an official Tensorflow product, it's one of the most reliable, maintained and well-built frameworks based on Tensorflow for deploying reinforcement learning agents. It includes the most popular & solid reinforcement learning algorithms, as well as their extensions. More info about the agents can be found here. Some of the supported algorithms are:

- Deep Q-Networks (DQN) + Extensions: Mutli-Step, Double DQN, Dueling Networks, C51
- REINFORCE
- Proximal Policy Optimization (PPO) with A2C, Epsilon Clipping, KL-Penalty, Entropy Regularization
- Soft-Actor Critic (SAC)
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Deep Deterministic Policy Gradient (DDPG)

It also prodides both Fully Connected + CNN/LSTM Layers. Customs layers can be added as well.

In this repository, I cover the rainbow DQN/C51 and the PPO algorithm. More information about the algorithms can be found here:
https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents

# RL-Lib
RLlib is an open-source library for reinforcement learning (RL), offering support for production-level, highly distributed RL workloads while maintaining unified and simple APIs for a large variety of industry applications. Whether you would like to train your agents in a multi-agent setup, purely from offline (historic) datasets, or using externally connected simulators, RLlib offers a simple solution for each of your decision making needs. Some of the supported algorithms are:

- Rainbow DQN
- APEX DQN
- Actor-Critic (A2C/A3C)
- DDPG/TD3
- APEX-DDPG
- IMPALA
- R2D2
- PPO/APPO
- AlphaZero
- Model-Based Meta-Policy-Optimization (MBMPO)

More info about the algorithms can be found here: https://docs.ray.io/en/latest/rllib-algorithms.html

# Future Work
1. Add frameworks: Google Dopamine, ACME.
2. Add more algorithms.
3. Add invalid action handling on custom environments.
