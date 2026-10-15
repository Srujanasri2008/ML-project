# Mathematical Wireless & Task Model

This document specifies the exact equations implemented in `src/semantic_noma/wireless_math.py`, `src/semantic_noma/channel_model.py`, and `src/semantic_noma/task_model.py`.

## 1. Large-Scale Path Loss and Small-Scale Fading

### 1.1 Free-Space Path Loss at Reference Distance
At reference distance $d_0 = 1$ m, carrier frequency $f_c = 3.5 \times 10^9$ Hz, and speed of light $c = 3.0 \times 10^8$ m/s:

$$PL_0 = 20 \log_{10}\left( \frac{4 \pi d_0 f_c}{c} \right)$$

Evaluating numerically:
$$PL_0 = 20 \log_{10}\left( \frac{4 \pi \cdot 1.0 \cdot 3.5 \times 10^9}{3.0 \times 10^8} \right) \approx 43.3229 \text{ dB}$$

### 1.2 User Path Loss
For user $i$ located at distance $d_i \in [30, 250]$ m with path-loss exponent $n = 3.5$:

$$PL_i = PL_0 + 10 n \log_{10}\left( \frac{d_i}{d_0} \right)$$

### 1.3 Channel Power Gain
The instantaneous channel power gain $g_i(t)$ combines large-scale path loss and complex circular symmetric Gaussian small-scale fading $h_i(t) \sim \mathcal{CN}(0, 1)$:

$$g_i(t) = 10^{-PL_i / 10} |h_i(t)|^2$$

Since $\mathbb{E}[|h_i(t)|^2] = 1$, the small-scale fading power follows an exponential distribution $\text{Exp}(1)$.

---

## 2. Receiver Noise Power

The thermal noise density $N_0 = -174$ dBm/Hz is converted to W/Hz by subtracting 30 dB:
$$N_{0,\mathrm{W/Hz}} = 10^{(-174 - 30)/10} = 10^{-20.4} \approx 3.98107 \times 10^{-21} \text{ W/Hz}$$

For subchannel $m$ with bandwidth $B_m = 180$ kHz and receiver noise figure $NF = 7$ dB ($10^{0.7} \approx 5.01187$):

$$N_m = 10^{(-174-30)/10} B_m 10^{NF/10} \approx 3.98107 \times 10^{-21} \times 180000 \times 5.01187 \approx 3.59146 \times 10^{-15} \text{ W}$$

---

## 3. Downlink NOMA Signal-to-Interference-plus-Noise Ratio (SINR)

For pair $g = (u_w, u_s)$ where $g_w \le g_s$:
- Transmit power for weak user: $p_w = \beta_g P_g$
- Transmit power for strong user: $p_s = (1 - \beta_g) P_g$
- With $\beta_g \in [0.5, 0.95]$ and $P_g = \eta_g \frac{P_{\max}}{M}$.

### 3.1 Weak User Direct Detection
The weak user detects its own signal directly while treating the strong user's signal as interference:

$$\gamma_w = \frac{p_w g_w}{p_s g_w + N_m}$$

### 3.2 Strong User Successive Interference Cancellation (SIC)
The strong user first decodes the weak user's signal:

$$\gamma_{w \to s} = \frac{p_w g_s}{p_s g_s + N_m}$$

Upon successful decoding and cancellation under ideal SIC, the strong user decodes its own signal interference-free:

$$\gamma_s = \frac{p_s g_s}{N_m}$$

---

## 4. Achievable Transmission Rates

To guarantee that the weak user's message is decodable at both receivers (weak user and strong user performing SIC):

$$R_w = B_m \min\left( \log_2(1 + \gamma_w), \log_2(1 + \gamma_{w \to s}) \right)$$

$$R_s = B_m \log_2(1 + \gamma_s)$$

Both rates are measured in bits per second (bps).

---

## 5. Airtime Allocation & Energy

Candidate pairs $\mathcal{G}_m$ sharing subchannel $m$ receive airtime fractions $\tau_g$:

$$\sum_{g \in \mathcal{G}_m} \tau_g \le 1, \quad 0 \le \tau_g \le 1$$

Delivered service bits in frame $t$:
$$c_i(t) = \tau_g T_f R_i(t)$$

Energy expended by pair $g$ during frame $t$:
$$E_g = \tau_g T_f P_g$$

Total frame energy:
$$E_t = T_f \sum_{g=1}^G \tau_g P_g$$

Normalized frame energy:
$$E_{\mathrm{norm}}(t) = \frac{E_t}{P_{\max} T_f} \in [0, 1]$$

---

## 6. Semantic Task Progress & Deadline Penalty

Useful service:
$$s_i(t) = \min(q_i(t), c_i(t))$$

Normalized progress for user $i$:
$$u_i(t) = w_i \frac{s_i(t)}{D_i}$$

Mean semantic utility across $K = 8$ users:
$$U_{\mathrm{sem}}(t) = \frac{1}{K} \sum_{i=1}^K u_i(t)$$

Deadline miss penalty $M_t$:
$$M_t = \frac{1}{K} \sum_{i \in \mathcal{D}_t} w_i \frac{q_i(t) - s_i(t)}{D_i}$$

where $\mathcal{D}_t$ is the set of tasks that reach deadline zero without complete delivery.
