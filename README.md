# Reinforcement Learning & Function Approximation Algorithms

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Environments-green.svg)](https://gymnasium.farama.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Optimization-blueviolet.svg)](https://optuna.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)

A comprehensive benchmark and implementation of fundamental Reinforcement Learning (RL), Dynamic Programming (DP), and Function Approximation algorithms developed at Sorbonne University (M.Sc. in AI & Robotics).

---

## 📌 Repository Modules

### 1. Dynamic Programming & Exact Planning (`01_dynamic_programming/`)
* **Value Iteration ($V$ and $Q$ functions):** Contraction mapping via Bellman Optimality Operators to achieve uniform convergence.
* **Policy Iteration:** Alternating policy evaluation and greedy policy improvement.
* **Generalized Policy Iteration (GPI):** Empirical study of truncated evaluation horizons ($K$-step updates) vs. full Bellman convergence.

### 2. Model-Free Tabular Reinforcement Learning (`02_tabular_rl/`)
* **Temporal Difference Learning [TD(0)]:** On-policy evaluation of state-value functions without transition probability knowledge.
* **Q-Learning (Off-Policy TD Control):** Tabular action-value iteration with Bellman optimality backups.
* **SARSA (On-Policy TD Control):** Explicit evaluation of next-state-action pairs under the executing behavioral distribution.
* **Exploration vs. Exploitation:** Empirical comparison of $\epsilon$-greedy selection against temperature-controlled Boltzmann Softmax policies.

### 3. Function Approximation & Kernel Methods (`03_function_approximation/`)
* **Linear Least Squares (LLS):** Normal equation formulation with augmented design matrices:
  $$\boldsymbol{\theta}^* = (\bar{\mathbf{X}}^\intercal \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^\intercal \mathbf{y}$$
* **Tikhonov / Ridge Regularization:** Weight penalty optimization preventing ill-conditioned inversions:
  $$\boldsymbol{\theta}^* = (\lambda \mathbf{I} + \bar{\mathbf{X}}^\intercal \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^\intercal \mathbf{y}$$
* **Radial Basis Function Networks (RBFNs):** Nonlinear continuous regression via multivariate Gaussian kernels and Gram matrix projections.

### 4. Actor-Critic with Bayesian Optimization (`04_actor_critic_optuna/`)
* **Actor-Critic Architecture:** Simultaneous parameterization of a categorical policy network (Actor) and a TD value estimator (Critic) in Gymnasium environments.
* **Hyperparameter Tuning Benchmark:** Comparing classical Grid Search with Bayesian Optimization via Tree-structured Parzen Estimators (TPE) in `Optuna`.
* **Statistical Validation:** Convergence rate and reward variance assessed across multi-seed runs using Welch's t-test.

---
