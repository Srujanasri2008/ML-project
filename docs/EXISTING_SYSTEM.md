# Existing System Architecture & Conventional Baselines

## 1. Traditional Wireless Access Architectures
In conventional cellular networks, resource allocation is typically optimized for bit-level performance under either purely orthogonal or non-orthogonal paradigms:

### 1.1 Pure OFDMA Systems
In conventional Orthogonal Frequency Division Multiple Access (OFDMA), subchannels are allocated exclusively to single users in time and frequency. While inter-user interference is completely eliminated, OFDMA suffers from poor spectral efficiency when user channel conditions are highly asymmetric, leaving weak-channel users starved of bandwidth or consuming excessive resources.

### 1.2 Pure NOMA Systems
Power-Domain Non-Orthogonal Multiple Access (NOMA) multiplexes multiple users on the same frequency resource by exploiting power differentials. Successive Interference Cancellation (SIC) enables the strong user to remove the weak user's signal. However, multiplexing more than two users per carrier creates excessive computational overhead and severe error propagation under practical transceivers.

### 1.3 Hybrid NOMA–OFDMA Systems
Hybrid NOMA–OFDMA strikes an optimal balance: it divides the spectrum into orthogonal subchannels and pairs exactly two users per subchannel. However, in existing systems:
- User pairing and subchannel assignments are typically static or heuristic.
- Power allocation is frequently solved via static fractional splits (e.g., fixed $\beta = 0.75$).
- Time-sharing across candidate pairs is often fixed to equal half-frames ($\tau = 0.5$).

---

## 2. Conventional Resource Allocation Approaches

Existing baseline approaches evaluated in this project include:

### 2.1 Equal Allocation Baseline
- **Mechanism:** Divides airtime equally ($\tau_g = 0.5$) between candidate pairs sharing each subchannel.
- **Power Strategy:** Operates at the full instantaneous power limit ($P_g = P_{\max}/M = 0.5$ W) regardless of channel fades or queue status.
- **Split Strategy:** Implements a fixed weak/strong split ($\beta = 0.75, 1 - \beta = 0.25$).
- **Limitations:** Highly energy-inefficient. Spends full transmit power even when a user needs only a tiny fraction of bits to complete its task, or when severe channel fading makes transmission wasteful.

### 2.2 Semantic-Priority Heuristic Baseline
- **Mechanism:** Dynamically ranks pairs according to user task importance $w_i$, remaining demand, and task urgency ($1 / \text{deadline}$).
- **Power Strategy:** Scales power utilization $\eta_g$ proportionally to pair priority relative to the highest-priority pair.
- **Split Strategy:** Biases NOMA power towards the user with higher urgency.
- **Limitations:** Lacks global foresight and long-term value estimation. While it handles urgent tasks effectively, it cannot optimize future queue transitions or balance discounted multi-step trade-offs.

### 2.3 Random Feasible Allocation Baseline
- **Mechanism:** Samples random continuous controls within the feasible domain.
- **Limitations:** High deadline-miss rate ($\approx 30\%$) and poor goodput ($1.54$ Mbps), demonstrating that uncoordinated allocation degrades network performance.

---

## 3. Why Existing Systems Fall Short for Semantic Communications
Conventional systems optimize solely for raw throughput (bps) or SNR:
1. They treat all transmitted bits as equally valuable, wasting power delivering irrelevant bits to users whose deadlines have already expired or whose tasks are unimportant.
2. They do not incorporate task deadlines or semantic importance weights directly into the physical layer control loop.
3. They fail to jointly optimize continuous power and fractional airtime under hard multi-constraint coupling.
