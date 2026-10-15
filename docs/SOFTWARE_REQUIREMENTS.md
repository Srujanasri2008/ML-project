# Software Requirements & Environment Specifications

## 1. System Platform
- **Operating System:** Windows 10/11 64-bit
- **Python Version:** Python 3.12 (compatible with 3.11–3.12)
- **Virtual Environment:** Dedicated `.venv` located strictly inside `semantic_noma_ofdma/`

## 2. Core Dependencies
- `torch==2.14.1+cpu`: CPU-only build installed from the PyTorch CPU repository
- `gymnasium==1.0.0`: Reinforcement learning environment API
- `stable-baselines3==2.9.0`: Stable SAC reinforcement learning algorithm
- `numpy==2.0.2`: Fast vectorized linear algebra and array computations
- `pandas==3.0.6`: Evaluation data serialization and metrics summarization
- `matplotlib==3.11.2`: Plot generation for training curves and comparative benchmarks
- `tensorboard==2.21.0`: RL training diagnostic logging
- `pytest==9.1.1`: Automated unit and integration testing suite

## 3. Optional Dashboard Dependency
- `streamlit>=1.32.0`: Interactive local browser dashboard (installed via `requirements-dashboard.txt` if requested)

## 4. Hardware and Computational Constraints
- **PyTorch Execution Device:** CPU
- **Maximum Numerical Threads:** 2 threads (`torch.set_num_threads(2)`)
- **Memory Footprint:** Under 1.5 GB RAM during execution
- **Disk Usage:** Project code and virtual environment $< 1.2$ GB (minimum 5 GB required at preflight)
- **Maximum Wall-Clock Runtime:** 600 seconds per standard training run
