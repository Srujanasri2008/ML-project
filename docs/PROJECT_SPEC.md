# System Specification: Semantic NOMA-OFDMA with SAC

## 1. Overview
This project implements a continuous reinforcement learning simulation for joint power and airtime allocation in a hybrid Non-Orthogonal Multiple Access (NOMA) and Orthogonal Frequency Division Multiple Access (OFDMA) wireless network serving semantic communication tasks.

## 2. Network Specifications
- **Number of Users:** $K = 8$
- **Number of NOMA Pairs:** $G = 4$ fixed pairs
  - Pair 0: Users {0, 1}
  - Pair 1: Users {2, 3}
  - Pair 2: Users {4, 5}
  - Pair 3: Users {6, 7}
- **Number of OFDMA Subchannels:** $M = 2$
  - Subchannel 0: Candidate Pairs {0, 1}
  - Subchannel 1: Candidate Pairs {2, 3}
- **Carrier Frequency:** $f_c = 3.5$ GHz
- **Subchannel Bandwidth:** $B_m = 180$ kHz
- **Total Instantaneous Power Cap:** $P_{\max} = 1.0$ W ($P_{\max}/M = 0.5$ W per subchannel)
- **Frame Duration:** $T_f = 0.01$ s (10 ms)
- **Episode Length:** $T_{\mathrm{ep}} = 50$ frames (0.5 s physical simulation per episode)
- **User Distances:** $d_i \in [30.0, 250.0]$ meters
- **Path-loss Exponent:** $n = 3.5$
- **Noise Model:** Thermal noise density $-174$ dBm/Hz with $NF = 7$ dB receiver noise figure.

## 3. Algorithm Specifications
- **Agent:** Soft Actor-Critic (SAC) continuous actor-critic
- **State Dimension:** Exactly 49 features
- **Action Dimension:** Exactly 12 continuous values in $[-1, 1]$
- **Feasibility:** Enforced by construction through analytical transformations
- **Execution Budget:** Up to 20,000 timesteps with 600s wall-clock cutoff and 2 CPU threads max.
