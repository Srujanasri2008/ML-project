# Problem Statement

## 1. Context
Next-generation cellular networks must support diverse intelligent devices whose transmissions convey task-oriented semantic information rather than arbitrary bitstreams. In hybrid NOMA–OFDMA architectures:
- Orthogonal subchannels ($M$) eliminate inter-subchannel interference.
- Non-orthogonal power-domain multiplexing within each pair allows simultaneous transmissions to a weak-channel user and a strong-channel user.
- Candidate pairs sharing a subchannel must time-share the frame duration $T_f$ via fractional airtime allocation $\tau_g$.

## 2. Objective Formulation
The network seeks to maximize cumulative discounted semantic utility while penalizing energy expenditure and task deadline expirations:

$$\max_{\{\eta_g(t), \beta_g(t), \tau_g(t)\}} \mathbb{E}\left[ \sum_{t=0}^{T_{\mathrm{ep}}-1} \gamma^t \left( U_{\mathrm{sem}}(t) - \lambda_E E_{\mathrm{norm}}(t) - \lambda_D M_t \right) \right]$$

subject to:
1. **Instantaneous Total Transmit Power Constraint:**
   $$\sum_{m=1}^M \max_{g \in \mathcal{G}_m} P_g(t) \le P_{\max}, \quad \forall t$$
   where $P_g(t) = \eta_g(t) \frac{P_{\max}}{M}$ with $0 \le \eta_g(t) \le 1$.

2. **NOMA Power Decomposition Constraint:**
   $$p_{w, g}(t) + p_{s, g}(t) = P_g(t), \quad 0.5 \le \beta_g(t) \le 0.95$$
   $$p_{w, g}(t) = \beta_g(t) P_g(t), \quad p_{s, g}(t) = (1 - \beta_g(t)) P_g(t)$$

3. **Subchannel Airtime Feasibility Constraint:**
   $$\sum_{g \in \mathcal{G}_m} \tau_g(t) \le 1, \quad 0 \le \tau_g(t) \le 1, \quad \forall m \in \{1, \dots, M\}$$

4. **Normalized Frame Energy Constraint:**
   $$E_t = T_f \sum_{g=1}^G \tau_g(t) P_g(t) \implies E_{\mathrm{norm}}(t) = \frac{E_t}{P_{\max} T_f} \in [0, 1]$$

5. **Semantic Progress Capping:**
   $$s_i(t) = \min(q_i(t), c_i(t))$$
   where $c_i(t) = \tau_g(t) T_f R_i(t)$ and $q_i(t)$ is remaining task demand.
