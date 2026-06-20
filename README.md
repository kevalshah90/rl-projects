# RL Projects

A collection of reinforcement learning experiments, implementations, and resources.

## Overview

This repository contains my journey into Reinforcement Learning (RL), including implementations of classic algorithms, experiments with modern techniques, and real-world applications.

---

## Core Topics

### Fundamentals
- **Markov Decision Processes (MDPs)** — State spaces, action spaces, transition probabilities, rewards
- **Bellman Equations** — Bellman expectation and optimality equations
- **Dynamic Programming** — Policy iteration, value iteration
- **Monte Carlo Methods** — First-visit vs. every-visit, MC prediction and control
- **Temporal Difference Learning** — TD(0), SARSA, Q-Learning

### Value-Based Methods
- **DQN (Deep Q-Networks)** — Experience replay, target networks, Double DQN
- **Rainbow** — Combining 6 DQN improvements
- **C51** — Distributional RL
- **QR-DQN** — Quantile Regression DQN
- **IQN** — Implicit Quantile Networks

### Policy-Based Methods
- **REINFORCE** — Monte Carlo policy gradients
- **Actor-Critic** — A2C, A3C
- **PPO (Proximal Policy Optimization)** — Clipped surrogate objective
- **TRPO** — Trust Region Policy Optimization
- **SAC (Soft Actor-Critic)** — Maximum entropy RL
- **TD3** — Twin Delayed DDPG

### Model-Based RL
- **Dyna-Q** — Integration of learning and planning
- **MBPO** — Model-Based Policy Optimization
- **PlaNet** — Deep planning network
- **Dreamer** — World models in latent space

### Advanced Topics
- **Multi-Agent RL** — Independent learning, centralized training with decentralized execution
- **Hierarchical RL** — Options framework, feudal networks
- **Offline RL (Batch RL)** — Learn from fixed datasets
- **Imitation Learning** — Behavioral cloning, inverse RL
- **Preference-Based RL** — Learning from human preferences
- **RLHF** — Reinforcement Learning from Human Feedback

### Exploration Strategies
- **ε-Greedy** — Basic exploration
- **UCB** — Upper Confidence Bound
- **Thompson Sampling** — Bayesian approach
- **Count-Based Exploration** — Pseudo-counts, hash-based
- **Curiosity-Driven** — Intrinsic motivation, ICM

---

## Learning Resources

### Courses
| Course | Institution | Level | Link |
|--------|-------------|-------|------|
| **CS285: Deep RL** | UC Berkeley | Graduate | [YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyMwkTCebZJnhLL) |
| **CS234: RL** | Stanford | Graduate | [Website](http://web.stanford.edu/class/cs234/) |
| **RL Specialization** | UAlberta (Coursera) | Beginner-Intermediate | [Coursera](https://www.coursera.org/specializations/reinforcement-learning) |
| **Introduction to RL** | DeepMind | Beginner | [YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0) |
| **Spinning Up in Deep RL** | OpenAI | Intermediate | [OpenAI](https://spinningup.openai.com/) |

### Books
| Book | Authors | Difficulty |
|------|---------|------------|
| **Reinforcement Learning: An Introduction** | Sutton & Barto | 🟢 Classic, must-read |
| **Deep RL Hands-On** | Max Lapan | 🟡 Practical, code-heavy |
| **Grokking Deep RL** | Miguel Morales | 🟢 Beginner-friendly |
| **Algorithms for RL** | Csaba Szepesvári | 🔴 Theoretical |
| **RL and Optimal Control** | Dimitri Bertsekas | 🔴 Advanced |

### Papers
| Paper | Why It Matters |
|-------|----------------|
| [DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) | Deep RL breakthrough |
| [A3C (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783) | Async actor-critic |
| [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) | Stable, simple, effective |
| [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905) | Sample-efficient continuous control |
| [Dreamer (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603) | World models in latent space |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | RLHF for LLMs |

### Prime-RL
**[Prime-RL](https://github.com/Prime-RL/Prime-RL)** — A resource for RL/LLM research and experimentation.

Implements methods for exploring the compute-efficiency frontier of RL algorithms for LLM training.

Key features:
- Efficient RL training for large language models
- Novel exploration strategies
- Tools for RLHF experimentation
- Research-grade implementations

### Key Libraries

| Library | Description | Use Case |
|---------|-------------|----------|
| [**Stable-Baselines3**](https://github.com/DLR-RM/stable-baselines3) | Reliable implementations of RL algorithms | Production, research |
| [**RLlib**](https://docs.ray.io/en/latest/rllib/index.html) | Scalable RL from Ray | Distributed training |
| [**TorchRL**](https://github.com/pytorch/rl) | PyTorch-native RL | PyTorch ecosystem |
| [**Tianshou**](https://github.com/thu-ml/tianshou) | Modular RL in PyTorch | Research, customization |
| [**CleanRL**](https://github.com/vwxyzjn/cleanrl) | Single-file implementations | Learning, reproduction |
| [**Gymnasium**](https://github.com/Farama-Foundation/Gymnasium) | Standardized environments | All RL experiments |
| [**PettingZoo**](https://github.com/Farama-Foundation/PettingZoo) | Multi-agent environments | MARL research |

---

## Project Ideas

### Classic Implementations
- [ ] CartPole with DQN
- [ ] MountainCar with SARSA
- [ ] LunarLander with PPO
- [ ] Breakout with Rainbow

### Continuous Control
- [ ] MuJoCo walker with SAC
- [ ] Robosuite manipulation tasks
- [ ] Hand dexterity with DAPG

### Multi-Agent
- [ ] Predator-Prey environment
- [ ] Cooperative/competitive game
- [ ] Communication emergence

### Real-World Applications
- [ ] Trading bot
- [ ] Resource scheduling
- [ ] Recommendation system
- [ ] Game playing agent

### Advanced
- [ ] Model-based agent (Dyna, MBPO)
- [ ] Intrinsic motivation implementation
- [ ] Offline RL from logged data
- [ ] RLHF pipeline for text generation

---

## Repository Structure

```
rl-projects/
├── classic/
│   ├── dqn/
│   ├── ppo/
│   └── sac/
├── model-based/
├── multi-agent/
├── offline-rl/
├── experiments/
└── utils/
```

---

## Getting Started

### Prerequisites
```bash
pip install torch numpy gymnasium
pip install stable-baselines3[extra]
```

### Quick Test
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

---

## Resources by Difficulty

| Level | Resources |
|-------|-----------|
| **Beginner** | Sutton & Barto Ch 1-6, UAlberta Course, Grokking Deep RL |
| **Intermediate** | CS285 Lectures, Spinning Up, SB3 Implementations |
| **Advanced** | Papers, MBPO/Dreamer, Multi-Agent RL, RLHF |
| **Research** | ArXiv daily, NeurICML/ICLR proceedings, OpenReview |

---

## Contributing

Feel free to add implementations, experiments, or resources!

## License

MIT License
