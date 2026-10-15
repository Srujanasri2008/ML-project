# Proposed Solution Architecture

## 1. System Overview
The proposed solution implements a complete simulation and continuous control framework composed of four modular subsystems:

```text
       ┌────────────────────────────────────────────────────────┐
       │                  Wireless Environment                  │
       │                                                        │
       │  ┌────────────────────┐      ┌──────────────────────┐  │
       │  │   Channel Model    │      │     Task Manager     │  │
       │  │ (Rayleigh + PL)    │      │(Demands & Deadlines) │  │
       │  └─────────┬──────────┘      └──────────┬───────────┘  │
       └────────────┼────────────────────────────┼──────────────┘
                    │ Current Gains              │ Queue State
                    ▼                            ▼
       ┌────────────────────────────────────────────────────────┐
       │               49-Dimensional State Vector              │
       │  (6 user features * 8 users + 1 episode horizon frac)  │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │            Continuous Soft Actor-Critic (SAC)          │
       │                Actor Network [128, 128]                │
       └────────────────────────────┬───────────────────────────┘
                                    │ Raw Actions [-1, 1]^12
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │          Analytical Feasibility Transformation         │
       │  eta_g in [0, 1]   beta_g in [0.5, 0.95]   tau_g in [0,1]│
       └────────────────────────────┬───────────────────────────┘
                                    │ Feasible Controls
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Downlink NOMA-OFDMA Allocator              │
       │  - Weak/Strong Identification  - NOMA Rates & SINR     │
       │  - Subchannel Airtime Sharing  - Delivered Bits        │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │               Reward & Task Queue Update               │
       │      r_t = U_sem(t) - lambda_E*E_norm - lambda_D*M_t   │
       └────────────────────────────────────────────────────────┘
```

## 2. Key Modules
1. **`channel_model.py`:** Generates Rayleigh fading power gains combined with large-scale log-distance attenuation for users at distances from 30 to 250 meters.
2. **`wireless_math.py`:** Evaluates thermal noise, NOMA weak/strong SINRs, successive interference cancellation rates, and transmission energies.
3. **`task_model.py`:** Manages task queues, delivery capping, completion latency recording, deadline decrements, and non-overlapping task replacements.
4. **`resource_allocator.py`:** Implements the closed-form feasibility mapping ensuring total instantaneous power $\le P_{\max}$ and subchannel airtime $\le 1.0$.
5. **`environment.py`:** Gymnasium standard environment exposing the 49-dimensional state and 12-dimensional continuous action space.
6. **`baselines.py`:** Equal allocation, semantic priority heuristic, and random feasible allocation benchmarks.
7. **`training.py` & `evaluation.py`:** SAC training loop with CPU thread limits, time-limit callbacks, validation checkpointing, and evaluation across 20 held-out seeds.
