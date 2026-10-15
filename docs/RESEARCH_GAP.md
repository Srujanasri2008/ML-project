# Research Gap & Scientific Framing

## 1. Precise Research Question
The core research question addressed by this project is:

> *Can an off-policy continuous Soft Actor-Critic (SAC) reinforcement learning agent effectively learn to jointly balance transmit power, intra-pair NOMA power splitting, and inter-pair subchannel airtime sharing to maximize semantic task delivery under strict physical constraints and heterogeneous user deadlines?*

## 2. Realistic Novelty Framing
In accordance with academic integrity guidelines, we explicitly state:
- Combining SAC, NOMA, OFDMA, and semantic task modeling is **not claimed as an entirely novel conceptual invention**. Each individual component exists in prior literature (see `docs/LITERATURE_REVIEW.md`).
- Explicit continuous airtime optimization within candidate pairs sharing an OFDMA subchannel has been explored in various queueing frameworks.

## 3. Explicit Contributions of this Implementation
The value of this work lies in the **concrete, reproducible software implementation and transparent empirical evaluation** of this formulation:
1. **Hard Feasibility by Construction:** Rather than relying on fragile soft penalty weights in the RL reward (which often lead to severe constraint violations during exploration), we derive and test an analytical transformation mapping $[-1, 1]^{12} \to (\eta, \beta, \tau)$ that guarantees $0 \le P_g \le P_{\max}/M$, $0.5 \le \beta_g \le 0.95$, and $\sum_{g \in \mathcal{G}_m} \tau_g \le 1.0$ at every single environment step.
2. **Transparent Task Proxy:** We avoid misleading claims of "AI understanding language" by explicitly documenting the task model as a synthetic semantic-priority-weighted progress proxy.
3. **Reproducible Comparative Protocol:** We evaluate the learned policy on 20 strictly held-out test scenarios against three distinct baselines (Equal, Semantic-Priority Heuristic, and Random Feasible Allocation) with multi-metric reporting (goodput, latency, energy efficiency, Jain's fairness index, and zero constraint violations).
