# Soft Actor-Critic-Based Semantic-Aware Power and Time Allocation for Hybrid NOMA–OFDMA Networks

A continuous deep reinforcement learning research project for joint power allocation and airtime scheduling in multi-user semantic-aware wireless communication networks.

---

## 1. Project Purpose & Research Scope

In modern wireless systems serving task-oriented applications, delivering raw bits is often secondary to completing semantic goals prior to strict deadlines. In a downlink hybrid Non-Orthogonal Multiple Access (NOMA) and Orthogonal Frequency Division Multiple Access (OFDMA) cell, multiple users share limited bandwidth and transmit power.

This project investigates whether a **Soft Actor-Critic (SAC)** continuous reinforcement learning policy can jointly optimize:
1. **Power utilization ($\eta_g$):** How much instantaneous power each NOMA pair transmits.
2. **NOMA power split ($\beta_g$):** The fraction of pair power allocated to the weak-channel user versus the strong-channel user to facilitate Successive Interference Cancellation (SIC).
3. **Airtime allocation ($\tau_g$):** The fraction of the frame duration assigned to candidate pairs sharing an OFDMA subchannel.

> 📖 **Looking for a quick VS Code guide?** See the dedicated [VSCODE_EXECUTION_GUIDE.md](file:///C:/Users/SRUJANA%20BANDARU/.gemini/antigravity/scratch/semantic_noma_ofdma/VSCODE_EXECUTION_GUIDE.md) for step-by-step instructions and copy-paste commands.  
> 🎓 **Preparing a presentation, viva, or interview?** Read the complete [PROJECT_PRESENTATION_AND_EXPLANATION_GUIDE.md](file:///C:/Users/SRUJANA%20BANDARU/.gemini/antigravity/scratch/semantic_noma_ofdma/PROJECT_PRESENTATION_AND_EXPLANATION_GUIDE.md) for word-for-word pitch scripts, Streamlit dashboard explanations, and Q&A answers.

---

## 2. What Soft Actor-Critic (SAC) Does

Soft Actor-Critic is an off-policy, actor-critic deep reinforcement learning algorithm based on maximum entropy RL. Unlike deterministic policies (e.g., DDPG) or discrete action spaces (e.g., DQN):
- SAC controls **continuous action spaces** ($a_t \in [-1, 1]^{12}$).
- SAC maximizes expected reward **plus policy entropy**, promoting exploration and preventing premature convergence to brittle sub-optimal policies.
- In this environment, SAC directly outputs continuous preferences for power and airtime.

### Clarification: Semantic Utility vs. Classification Accuracy
> **IMPORTANT SCIENTIFIC DISTINCTION:**  
> This is a continuous resource-allocation project, **not a supervised image or text classification task**.  
> The agent outputs continuous wireless transmission parameters. Therefore, **metrics like accuracy, precision, recall, and F1-score are not applicable**.  
> The "semantic utility" $U_{\mathrm{sem}}$ is a priority-weighted progress proxy calculated from useful delivered bits, original demand, and task importance weights $w_i \in [0.2, 1.0]$. It does not represent deep language parsing or image classification accuracy.

---

## 3. Network & Wireless Model Summary

| Parameter | Value | Description |
|---|---|---|
| **Users ($K$)** | 8 | Grouped into 4 fixed NOMA pairs |
| **OFDMA Subchannels ($M$)** | 2 | 2 candidate pairs time-sharing each subchannel |
| **Total Max Power ($P_{\max}$)** | 1.0 W | Subchannel power budget $\le 0.5$ W |
| **Subchannel Bandwidth ($B_m$)** | 180 kHz | Standard LTE/5G subcarrier block |
| **Frame Duration ($T_f$)** | 0.01 s | 10 ms transmission frame |
| **Episode Horizon** | 50 frames | 0.5 seconds physical simulation |
| **Carrier Frequency ($f_c$)** | 3.5 GHz | Mid-band 5G frequency |
| **Path Loss Model** | Log-distance | $PL_0 = 43.32$ dB @ 1m, exponent $n = 3.5$ |
| **User Distances** | 30–250 m | Linearly distributed across the cell |
| **Fading Model** | Rayleigh | Complex Gaussian $h_i(t) \sim \mathcal{CN}(0, 1)$ |
| **Noise Model** | Thermal + NF | $-174$ dBm/Hz with $NF = 7$ dB ($N_m \approx 3.59 \times 10^{-15}$ W) |

---

## 4. Reinforcement Learning Design

- **Observation Space (49 Dimensions):**  
  For each of the 8 users: normalized log channel gain, normalized distance, remaining demand ratio ($q_i/D_i$), normalized original demand, semantic weight ($w_i$), and remaining deadline ratio.  
  Plus 1 global feature: remaining fraction of the episode.
- **Action Space (12 Continuous Values):**  
  For each of the 4 pairs: $[a_{\eta, g}, a_{\beta, g}, a_{\tau, g}] \in [-1, 1]^3$.
- **Hard Feasibility Enforcement by Construction:**  
  Analytical mappings guarantee $0 \le P_g \le P_{\max}/M$, $0.5 \le \beta_g \le 0.95$, and $\sum_{g \in \mathcal{G}_m} \tau_g \le 1.0$ at every step. Zero constraint violations are produced.
- **Reward Function:**  
  $$r_t = U_{\mathrm{sem}}(t) - \lambda_E E_{\mathrm{norm}}(t) - \lambda_D M_t$$  
  with default weights $\lambda_E = 0.01$ and $\lambda_D = 0.5$.

---

## 5. Software Requirements & Quickstart

### Prerequisites
- Python 3.12 (or 3.11) on Windows
- PowerShell 5.1+
- At least 5 GB free disk space

### Installation (One-Time Setup)
```powershell
# Open terminal in project directory
cd "C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma"

# Run preflight system check
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1

# Create virtual environment
py -3.12 -m venv .venv

# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install PyTorch CPU
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Install package in editable mode
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

---

## 6. How to Run

### Run Automated Tests (25 Test Cases)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Run Smoke Training (1,000 Timesteps)
```powershell
.\.venv\Scripts\python.exe run_all.py --smoke
```

### Run Standard Training & Evaluation (20,000 Timesteps)
```powershell
.\.venv\Scripts\python.exe run_all.py --timesteps 20000 --max-runtime-seconds 600 --eval-episodes 20
```

### Evaluate Checkpoint Without Retraining
```powershell
.\.venv\Scripts\python.exe run_all.py --skip-train --checkpoint ".\runs\run_20261014_192552\checkpoints\best_model.zip" --eval-episodes 20
```

### (Optional) Interactive Dashboard
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

---

## 7. Experimental Results (20 Held-Out Scenarios)

Evaluated across seeds 2000–2019:

| Metric | SAC (Proposed) | Equal Allocation | Semantic Priority | Random Allocation |
|---|---|---|---|---|
| **Episode Reward** | $12.15 \pm 0.35$ | $13.00 \pm 0.36$ | $12.23 \pm 0.34$ | $8.96 \pm 0.37$ |
| **Semantic Utility** | $0.2561 \pm 0.0064$ | $0.2760 \pm 0.0073$ | $0.2555 \pm 0.0072$ | $0.1990 \pm 0.0070$ |
| **Goodput** | $2.02$ Mbps | $2.18$ Mbps | $2.00$ Mbps | $1.54$ Mbps |
| **Task Completion Rate** | $78.8\%$ | $82.5\%$ | $81.9\%$ | $65.1\%$ |
| **Deadline Miss Rate** | $16.9\%$ | $13.5\%$ | $13.6\%$ | $29.8\%$ |
| **Total Energy** | **$0.257$ J** | $0.500$ J | $0.366$ J | $0.206$ J |
| **Energy Efficiency** | **$3.92 \times 10^6$ bits/J** | $2.18 \times 10^6$ bits/J | $2.73 \times 10^6$ bits/J | $3.75 \times 10^6$ bits/J |
| **Jain's Fairness** | $0.794$ | $0.833$ | $0.824$ | $0.814$ |
| **Constraint Violations** | **0.000000** | **0.000000** | **0.000000** | **0.000000** |

**Key Takeaway:** SAC learned to save nearly $50\%$ of transmit energy while delivering $92.5\%$ of maximum goodput, resulting in **$+79.8\%$ higher energy efficiency** than brute-force equal allocation.

---

## 8. Saved Artifacts Directory Structure

Each experiment run generates its own isolated directory under `runs/`:
- `run_config.json`: Run parameters and network settings.
- `training_log.csv`: Per-episode rewards and penalties during training.
- `evaluation_summary.csv`: Summary statistics (mean, std, min, max) for all policies.
- `per_seed_metrics.csv`: Detailed episode metrics for every test seed.
- `run_summary.json`: Top-level run summary with checkpoint links.
- `checkpoints/best_model.zip`: Validation-best model checkpoint.
- `checkpoints/final_model.zip`: Final trained model checkpoint.
- Visualization plots (`.png`):
  - `training_reward.png`: Training curve over timesteps.
  - `reward_components.png`: Semantic utility, energy, and deadline penalties.
  - `semantic_utility_comparison.png`: Policy comparison bar chart.
  - `throughput_comparison.png`: Goodput and sum rate comparison.
  - `latency_comparison.png`: Latency and deadline-miss rate comparison.
  - `energy_efficiency_comparison.png`: Bits per Joule comparison.
  - `fairness_comparison.png`: Jain's index comparison.
  - `power_allocation_example.png`: Dynamic power allocation across pairs.
  - `airtime_allocation_example.png`: Dynamic airtime sharing across pairs.

---

## 9. Limitations & Future Extensions
- **Ideal SIC Assumption:** Residual interference and imperfect CSI are not modeled.
- **Fixed User Pairing:** Dynamic pairing based on channel correlation could yield further gains.
- **Independent Fading:** Correlated temporal fading (Doppler/Jakes) is left as a future recurrent RL extension.
- **Synthetic Task Model:** Future extensions may integrate neural semantic encoders (e.g., DeepSC) for end-to-end semantic transmission.
