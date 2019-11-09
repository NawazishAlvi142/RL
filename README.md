# Reinforcement Learning in TensorFlow

```
TF-Agents : A library for Reinforcement Learning in TensorFlow
https://github.com/tensorflow/agents

官方範例網址
https://github.com/tensorflow/agents
```
```
Applied Reinforcement Learning: Playing Doom with TF-Agents and PPO

```
```
TensorFlow Agents: Efficient Batched Reinforcement Learning in TensorFlow
Danijar Hafner, James Davidson, Vincent Vanhoucke
(Submitted on 8 Sep 2017 (v1), last revised 31 Oct 2018 (this version, v2))

https://arxiv.org/abs/1709.02878

We introduce TensorFlow Agents, an efficient infrastructure paradigm for 
building parallel reinforcement learning algorithms in TensorFlow. 

We simulate multiple environments in parallel, and group them to perform the neural network computation 
on a batch rather than individual observations. 

This allows the TensorFlow execution engine to parallelize computation, 
without the need for manual synchronization. 

Environments are stepped in separate Python processes to progress them in parallel 
without interference of the global interpreter lock. 

As part of this project, we introduce BatchPPO, 
an efficient implementation of the proximal policy optimization algorithm. 

By open sourcing TensorFlow Agents, 
we hope to provide a flexible starting point for future projects that accelerates future research in the field.
```
### YOUTUBE頻道
```
Reinforcement Learning in TensorFlow with TF-Agents (TF Dev Summit '19)
https://www.youtube.com/watch?v=-TTziY7EmUA

TF-Agents: Reinforcement Learning (TensorFlow Meets)
https://www.youtube.com/watch?v=a_OfZoF4IYc

TF-Agents: A Flexible Reinforcement Learning Library for TensorFlow (Google I/O'19)
https://www.youtube.com/watch?v=tAOApRQAgpc

TF-Agents: A Flexible Reinforcement Learning Library for TensorFlow (Google I/O'19)
https://www.youtube.com/watch?v=tAOApRQAgpc
```
```
TensorFlow Tutorial #16 Reinforcement Learning[有點舊]
https://www.youtube.com/watch?v=Vz5l886eptw

Hvass-Labs/TensorFlow-Tutorials
https://github.com/Hvass-Labs/TensorFlow-Tutorials

Transfer Learning (Notebook) (Google Colab)
Adversarial Examples (Notebook) (Google Colab)
Adversarial Noise for MNIST (Notebook) (Google Colab)

DeepDream (Notebook) (Google Colab)

Style Transfer (Notebook) (Google Colab)

Reinforcement Learning (Notebook) (Google Colab)

Estimator API (Notebook) (Google Colab)
...........
TFRecords & Dataset API (Notebook) (Google Colab)
Natural Language Processing (Notebook) (Google Colab)
Machine Translation (Notebook) (Google Colab)
Image Captioning (Notebook) (Google Colab)
Time-Series Prediction (Notebook) (Google Colab)
```


### 快速入門Tutorials
```
https://github.com/tensorflow/agents/tree/master/tf_agents/colabs

0_intro_rl.ipynb	
1_dqn_tutorial.ipynb	
2_environments_tutorial.ipynb	
3_policies_tutorial.ipynb	
4_drivers_tutorial.ipynb
5_replay_buffers_tutorial.ipynb
6_reinforce_tutorial.ipynb
7_SAC_minitaur_tutorial.ipynb
8_networks_tutorial.ipynb
9_c51_tutorial.ipynb

© 2019 GitHub, Inc.
```

### algorithms available under TF-Agents
```
Currently the following algorithms are available under TF-Agents:

DQN: Human level control through deep reinforcement learning Mnih et al., 2015
DDQN: Deep Reinforcement Learning with Double Q-learning Hasselt et al., 2015
DDPG: Continuous control with deep reinforcement learning Lillicrap et al., 2015
TD3: Addressing Function Approximation Error in Actor-Critic Methods Fujimoto et al., 2018
REINFORCE: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning Williams, 1992
PPO: Proximal Policy Optimization Algorithms Schulman et al., 2017
SAC: Soft Actor Critic Haarnoja et al., 2018
```
# DRL-Tutorial
```
DRL-Tutorial
Deep Reinforcement Learning Tutorial Site for PLDI 2019
https://ai-vidya.github.io/DRL-Tutorial/

```

# TRFL[17 OCT 2018]
```
TRFL (pronounced "truffle") is a library built on top of TensorFlow that exposes 
several useful building blocks for implementing Reinforcement Learning agents.

https://github.com/deepmind/trfl

pip install trfl

https://deepmind.com/blog/article/trfl
```
