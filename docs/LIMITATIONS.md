# Limitations & Modeling Assumptions

This document explicitly details the assumptions, simplifications, and limitations of the simulated environment.

## 1. Synthetic Semantic-Utility Modeling
- The simulation employs a synthetic task-utility model where each user task has an assigned scalar importance weight $w_i \in [0.2, 1.0]$ and bits demand $D_i$.
- **Not a Neural NLP or Vision Model:** The system does not execute deep semantic autoencoders, transformer language models, or image feature extractors. The semantic utility $U_{\mathrm{sem}}$ serves as a priority-weighted transmission proxy and should not be confused with image classification accuracy, BLEU score, or semantic similarity metrics.

## 2. Ideal Successive Interference Cancellation (SIC)
- The wireless model assumes ideal SIC where the strong user cancels the weak user's signal without residual decoding error or error propagation.
- In practical transceivers, non-ideal channel state information (CSI), hardware impairments, and quantization errors leave residual interference that degrades the strong user's effective SINR.

## 3. Fixed NOMA User Pairing and Subchannel Assignment
- The four NOMA pairs and their assignment to the two OFDMA subchannels are held static throughout each episode.
- Dynamic user pairing based on instantaneous channel correlation and dynamic subcarrier allocation are viable future extensions but were excluded to isolate the joint continuous power and airtime problem.

## 4. Independent Frame-by-Frame Fading Model
- Small-scale fading is regenerated independently from frame to frame ($h_i(t) \sim \mathcal{CN}(0, 1)$).
- A continuous-time Doppler or Jakes fading model with temporal channel correlation would introduce time-series dependencies that could be leveraged by recurrent policy architectures (e.g., LSTM-SAC).

## 5. Sim-to-Real Transfer Limitations
- While simulation enables reproducible, risk-free RL experimentation, physical wireless networks exhibit complex interference from adjacent cells, synchronization errors, feedback delays in CSI reporting, and non-stationary traffic queues.
