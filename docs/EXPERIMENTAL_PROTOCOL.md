# Experimental Protocol & Evaluation Methodology

## 1. Experimental Integrity & Separation
To prevent data leakage and cherry-picking, random seeds are strictly partitioned into three independent groups:
1. **Training Environment Seed:** Seed `42` is used for all training environment dynamics and agent initialization.
2. **Validation Seeds:** Five fixed seeds `[1000, 1001, 1002, 1003, 1004]` are evaluated every 2,000 timesteps during training. The checkpoint achieving the highest validation mean reward is saved as `best_model.zip`.
3. **Held-Out Test Seeds:** Twenty distinct seeds `[2000, 2001, ..., 2019]` are reserved exclusively for the final evaluation of SAC and all baselines. Neither training updates nor checkpoint selection uses these seeds.

## 2. Compared Policies
Every policy is evaluated over the identical 20 held-out test scenarios with matching channel trajectories and task arrival distributions:
- **SAC (Proposed):** The best validation checkpoint loaded from `best_model.zip` executing deterministic continuous inference.
- **Equal Allocation (Baseline A):** Equal airtime sharing ($\tau_g = 0.5$ per pair on each subchannel), maximum continuous power utilization ($\eta_g = 1.0$), and fixed NOMA power split ($\beta_g = 0.75$).
- **Semantic Priority (Baseline B):** Deterministic heuristic allocating power utilization and airtime preference proportionally to user task urgency and semantic importance weights.
- **Random Allocation (Baseline C):** Uniformly sampled feasible actions using an independent random number generator.

## 3. Evaluated Metrics
- **Mean Semantic Utility ($U_{\mathrm{sem}}$):** Frame-averaged task progress proxy weighted by importance.
- **Goodput (bps):** Total useful delivered bits divided by the total simulated time ($T_{\mathrm{ep}} \times T_f = 0.5$ s).
- **Average Sum Rate (bps):** Mean sum of achievable transmission rates across users.
- **Task Completion Rate:** Completed tasks divided by total generated tasks.
- **Deadline-Miss Rate:** Expired incomplete tasks divided by total generated tasks.
- **Average Completed-Task Latency:** Average duration from task arrival to completion (frames).
- **Total Energy (Joules):** Cumulative simulated transmit energy.
- **Energy Efficiency (bits / Joule):** Total useful delivered bits per Joule of expended energy.
- **Jain's Fairness Index:** Measures uniformity of useful service delivery across all 8 users.
- **Constraint Violations:** Maximum observed violation of power limits and airtime budgets (strictly verified as 0.0).
