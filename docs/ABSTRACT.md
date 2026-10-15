# Research Abstract

**Soft Actor-Critic-Based Semantic-Aware Power and Time Allocation for Hybrid NOMA–OFDMA Networks**

### Abstract
Semantic communication represents a paradigm shift from traditional bit-level transmission toward meaning- and task-oriented wireless delivery. In multi-user wireless networks with heterogeneous task urgencies, varying channel fading, and stringent energy constraints, traditional equal or purely channel-dependent resource allocation fails to prioritize critical semantic deliverables. This project formulates and evaluates a continuous deep reinforcement learning (DRL) framework using Soft Actor-Critic (SAC) to jointly optimize power allocation and airtime sharing in a downlink hybrid Non-Orthogonal Multiple Access (NOMA) and Orthogonal Frequency Division Multiple Access (OFDMA) network.

Eight wireless users grouped into four fixed NOMA pairs across two orthogonal subchannels are simulated over Rayleigh fading channels with a log-distance path-loss model. Each user generates synthetic tasks characterized by original demand (2,000–10,000 bits), finite frame deadlines (2–6 frames), and semantic importance weights ($w_i \in [0.2, 1.0]$). A 49-dimensional normalized observation vector informs a 12-dimensional continuous actor network. To guarantee physical feasibility without relying on soft penalty tuning, analytical transformations enforce subchannel airtime budgets ($\sum_{g} \tau_g \le 1$) and instantaneous power limits ($\le P_{\max}$) by construction.

The trained SAC policy is evaluated on 20 held-out test scenarios and compared against equal allocation, a deterministic semantic-priority heuristic, and random feasible allocation. Empirical results demonstrate that the proposed joint formulation achieves zero constraint violations, minimizes deadline-miss penalties, and balances semantic task completion with transmission energy efficiency.
