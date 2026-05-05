# MSDS684: Reinforcement Learning

A comprehensive coursework repository for **MSDS 684: Reinforcement Learning**. This repository contains weekly lab assignments, notebooks, and implementations covering core reinforcement learning topics.

## Week 1: The Multi-Armed Bandit Problem and MDP Foundations

- Understand the fundamental RL problem through the bandit framework
- Implement and compare basic exploration strategies
- Formalize sequential decision problems as MDPs
- Map theoretical RL concepts (states, actions, rewards, transitions) to Gymnasium API
- Write agents that interact with standard RL environments

## Week 2: Dynamic Programming on GridWorld

- Implement policy iteration and value iteration on a custom GridWorld environment
- Build a configurable GridWorld with rewards, obstacles, terminal states, and stochastic transitions
- Compare synchronous and in-place update modes for both DP algorithms
- Visualize value functions as heatmaps, policies as quiver plots, and convergence curves

## Week 3: First-Visit Monte Carlo Control for Blackjack

- Implement first-visit Monte Carlo control with epsilon-soft policies on Gymnasium's Blackjack-v1
- Explore five epsilon decay strategies: linear, exponential, cosine, step, and adaptive
- Compare trained agent against naive baselines (random, always-hit, always-stick) and rules-based strategies
- Test three hypotheses: fixed epsilon optimality, scheduled decay, and adaptive decay effectiveness

## Week 4: SARSA vs Q-Learning on CliffWalking

- Implement SARSA (on-policy) and Q-learning (off-policy) on CliffWalking-v1 across 30 seeds
- Sweep over learning rates and epsilon schedules to identify optimal hyperparameter configurations
- Demonstrate the exploration-safety trade-off: SARSA learns a safe inland path; Q-learning learns the riskier cliff-edge shortcut
- Visualize policy comparisons, value function heatmaps, and trajectory overlays

## Week 5: Function Approximation with Semi-Gradient SARSA and Tile Coding

- Implement semi-gradient SARSA with tile coding on MountainCar-v0 (continuous state space)
- Explore five tiling configurations (4-16 tilings; 8x8-16x16 tiles) to study the resolution-vs-cost trade-off
- Find that tile resolution drives outcome quality while tiling count drives computational cost
- Achieve 91.7% success rate with the best configuration (16 tilings, 8x8 tiles) within a 2,500-episode budget

## Week 6: One-Step Actor-Critic on LunarLanderContinuous

- Implement a one-step actor-critic agent with separate PyTorch neural networks for actor (Gaussian policy) and critic (state-value)
- Train 30 independent agents across 1,000 episodes; aggregate via ensemble mean-action voting
- Demonstrate semi-gradient policy optimization and one-step TD updates on a continuous action space
- Analyze return distributions, policy entropy, and TD error to characterize training stability

## Week 7: Planning and Learning Integration (Dyna-Q Variants) on Taxi-v3

- Implement four algorithms on Taxi-v3: Q-learning (model-free baseline), Dyna-Q (n = 5, 10, 50 planning steps), Dyna-Q+, and prioritized sweeping
- Build a tabular world model enabling off-policy planning; track visitation times for the Dyna-Q+ exploration bonus
- Demonstrate sample efficiency gains from planning and robustness to dynamic environment changes
- Compare model-based vs. model-free trade-offs: computational cost, convergence speed, and sensitivity to model error

## Week 8: Modern Deep RL Exploration

- Train DQN on LunarLander-v3 using Stable-Baselines3, baselined on the rl-baselines3-zoo configuration
- Run hyperparameter sweeps across learning rate, exploration fraction, and target update interval across 20 seeds
- Ablate experience replay and target network to isolate each component's contribution
- Compare DQN against a one-step actor-critic baseline from Week 6
- Analyze performance distributions to characterize seed variance and evaluate reliability vs. expected return