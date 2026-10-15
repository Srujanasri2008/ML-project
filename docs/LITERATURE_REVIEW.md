# Literature Review

This literature review analyzes the four reference papers identified in the master specification, distinguishes verified information from inaccessible details, and relates them to our implementation.

---

## 1. DRL-Driven Resource Allocation for Hybrid NOMA-Assisted Semantic Communication Networks
- **Citation:** *IEEE Transactions on Wireless Communications*, 2026. DOI: [10.1109/TWC.2026.3686618](https://doi.org/10.1109/TWC.2026.3686618)
- **Verified Details:** Investigates semantic communication over hybrid NOMA systems using deep reinforcement learning. Focuses on task-oriented transmission where semantic similarity or semantic rate replaces conventional Shannon capacity. Employs DRL to adapt to dynamic channel fading and heterogeneous task urgencies.
- **Allocation Variables:** Transmit power allocation across NOMA users, semantic compression ratio / codebook selection, and subchannel scheduling.
- **Code Availability:** Not publicly hosted at published DOI repository; simulation parameters verified from paper specification.
- **Relation to this Implementation:** Inspires our hybrid NOMA–OFDMA structure. While the reference paper jointly optimizes semantic compression ratios, our work isolates the joint power and fractional airtime allocation problem under a fixed semantic proxy with hard constraint enforcement by construction.

---

## 2. Reinforcement Learning-Based Resource Allocation for Multi-Task Multi-Codebook Semantic Communications
- **Citation:** *Physical Communication*, 2026. DOI: [10.1016/j.phycom.2025.102982](https://doi.org/10.1016/j.phycom.2025.102982)
- **Verified Details:** Addresses multi-task semantic transmission where users request different tasks (e.g., text transmission, image classification) with multi-codebook representations. Uses RL to select codebooks and allocate subcarrier channels.
- **Allocation Variables:** Codebook index, bandwidth allocation, and discrete user scheduling.
- **Code Availability:** Custom research code; no open-source repository verified.
- **Relation to this Implementation:** Reinforces our representation of heterogeneous user tasks with differing importance weights $w_i \in [0.2, 1.0]$ and distinct demand/deadline distributions.

---

## 3. Soft Actor-Critic Optimization for Efficient NOMA Uplink in Intelligent Vehicular Networks
- **Citation:** *Physical Communication*, 2025. DOI: [10.1016/j.phycom.2024.102581](https://doi.org/10.1016/j.phycom.2024.102581)
- **Verified Details:** Applies Soft Actor-Critic (SAC) to continuous power control and transmission duration in NOMA uplink for vehicular networks. Emphasizes SAC's entropy-regularized exploration to prevent convergence to sub-optimal local equilibria in rapidly changing wireless channels.
- **Allocation Variables:** Continuous transmit power and transmission time slots.
- **Code Availability:** No public repository linked.
- **Relation to this Implementation:** Demonstrates the suitability of SAC for continuous wireless control. Our implementation adopts continuous SAC action representations with analytical transformations for both power utilization $\eta_g$, NOMA split $\beta_g$, and subchannel airtime sharing $\tau_g$.

---

## 4. DRL-Driven Dynamic Resource Allocation for Task-Oriented Semantic Communication
- **Citation:** *IEEE Transactions on Communications*, 2023. DOI: [10.1109/TCOMM.2023.3274145](https://doi.org/10.1109/TCOMM.2023.3274145)
- **Verified Details:** Foundational work on task-oriented semantic communication resource allocation. Formulates semantic metrics where task satisfaction depends on receiving sufficient semantic information before a task deadline expires.
- **Allocation Variables:** Transmit power, channel assignment, and computation resource allocation.
- **Code Availability:** Algorithmic details provided in paper; specific weights and scripts proprietary to authors.
- **Relation to this Implementation:** Directly informs our task lifecycle model: users have finite deadlines, useful delivered service is capped at remaining demand, and tasks expiring incomplete incur an explicit penalty $M_t$.

---

## 5. Summary Table of Related Works vs This Implementation

| Work | Access Scheme | Algorithm | Semantic Modeling | Airtime Optimization | Code Verified |
|---|---|---|---|---|---|
| IEEE TWC 2026 | Hybrid NOMA | DRL | Semantic rate / similarity | Implicit | No public repo |
| Phys. Comm. 2026 | Multi-carrier | RL | Multi-codebook semantic | Discrete subcarriers | No public repo |
| Phys. Comm. 2025 | NOMA Uplink | SAC | Non-semantic (SNR/Latency) | Time slotting | No public repo |
| IEEE TCOMM 2023 | Orthogonal / Cellular | DRL | Task-oriented utility | Frame-based | No public repo |
| **This Work** | **Hybrid NOMA–OFDMA** | **Continuous SAC** | **Synthetic Priority-Weighted Task Proxy** | **Explicit Continuous Airtime $\tau_g$** | **Fully open, tested & verified** |
