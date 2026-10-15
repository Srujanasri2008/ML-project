# SAC Methodology & RL Architecture

## 1. Soft Actor-Critic Overview
Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework. SAC maximizes a trade-off between expected discounted cumulative reward and policy entropy:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right) \right]$$

where $\alpha$ is the entropy temperature parameter controlling exploration and policy stochasticity.

## 2. Observation Space (49 Dimensions)
The observation vector $s_t \in \mathbb{R}^{49}$ is strictly defined and normalized as follows:

For each user $i \in \{0, \dots, 7\}$ (6 features per user = 48 values):
1. **Normalized Log Channel Gain:** $\text{clip}\left(\frac{\log_{10}(g_i(t)) + 10.0}{5.0}, -3.0, 3.0\right)$
2. **Normalized User Distance:** $\frac{d_i - d_{\min}}{d_{\max} - d_{\min}} \in [0, 1]$
3. **Remaining Demand Ratio:** $\frac{q_i(t)}{D_i} \in [0, 1]$
4. **Normalized Original Demand:** $\frac{D_i}{D_{\max}} \in [0, 1]$
5. **Semantic Importance Weight:** $w_i \in [0.2, 1.0]$
6. **Normalized Remaining Deadline:** $\frac{\text{deadline}_i(t)}{\text{deadline}_{\max}} \in [0, 1]$

Global 49th feature:
49. **Remaining Fraction of Current Episode:** $\frac{T_{\mathrm{ep}} - t}{T_{\mathrm{ep}}} \in [0, 1]$

## 3. Action Space (12 Continuous Dimensions)
The continuous action space $a_t \in [-1, 1]^{12}$ represents 3 raw controls per pair across 4 pairs:
- $a_{\eta, g} \in [-1, 1]$: Raw power utilization preference
- $a_{\beta, g} \in [-1, 1]$: Raw NOMA power split preference
- $a_{\tau, g} \in [-1, 1]$: Raw airtime preference

### Analytical Transformation (Constraint Enforcement by Construction)
To eliminate invalid states, raw actions are deterministically mapped to the feasible domain:

$$\eta_g = \frac{a_{\eta, g} + 1}{2} \in [0, 1]$$

$$\beta_g = 0.5 + 0.45 \cdot \frac{a_{\beta, g} + 1}{2} \in [0.5, 0.95]$$

$$z_g = \frac{a_{\tau, g} + 1}{2} \in [0, 1]$$

For candidate pairs $\mathcal{G}_m$ assigned to subchannel $m$:

$$\tau_g = \frac{z_g}{\max\left(1.0, \sum_{j \in \mathcal{G}_m} z_j\right)}$$

This transformation guarantees:
- $0 \le \tau_g \le 1$
- $\sum_{g \in \mathcal{G}_m} \tau_g \le 1$
- Handles the degenerate zero-preference case without division by zero ($\tau_g = 0$)
- Requires no clipping heuristics or constraint penalties in the reward function.

## 4. Hyperparameter Configuration

| Parameter | Value | Description |
|---|---|---|
| Policy Architecture | Multi-Layer Perceptron (MlpPolicy) | Fully-connected neural network |
| Hidden Layers | [128, 128] | Two hidden layers with ReLU activations |
| Learning Rate | $3 \times 10^{-4}$ | Adam optimizer learning rate |
| Discount Factor $\gamma$ | 0.99 | Temporal discount factor |
| Replay Buffer Size | 25,000 | Maximum transitions stored |
| Learning Starts | 500 steps | Initial uniform exploration steps before gradient updates |
| Batch Size | 128 | Mini-batch size per update |
| Training Frequency | Every 4 environment steps | Update trigger frequency |
| Gradient Steps | 1 per update | Gradient descent steps per train step |
| Target Entropy | Automatic | SB3 default heuristic ($-\dim(A) = -12$) |
| Device | CPU | CPU-only execution |
| Max CPU Threads | 2 | PyTorch and OpenBLAS thread cap |
