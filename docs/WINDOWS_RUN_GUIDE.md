# Windows Execution & Rerun Guide

Tested PowerShell commands to run, test, and evaluate the project on a Windows computer.

---

## 1. Prerequisites & Environment Setup

Open PowerShell in the project directory:
```powershell
Set-Location "C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma"
```

Verify system requirements and disk space (at least 5 GB free):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

If the virtual environment `.venv` does not already exist, create and install it once:
```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv .venv

# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install CPU-only PyTorch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Install package in editable mode without reinstalling dependencies
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

> **Note:** Once `.venv` is created, do NOT reinstall packages every time. You can directly run tests and experiments using `.\.venv\Scripts\python.exe`.

---

## 2. Running Automated Tests

Run the full pytest suite (25 test cases covering constraints, wireless equations, Gymnasium API, checkpoint loading, and reproducibility):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 3. Running Smoke-Training (1,000 Timesteps)

Execute a fast smoke test to verify training, validation, checkpoint saving, evaluation, and plot generation:

```powershell
.\.venv\Scripts\python.exe run_all.py --smoke
```

---

## 4. Running Standard Training & Evaluation (20,000 Timesteps)

Run standard SAC training, periodic validation across 5 seeds, and held-out evaluation across 20 test scenarios:

```powershell
.\.venv\Scripts\python.exe run_all.py --timesteps 20000 --max-runtime-seconds 600 --eval-episodes 20
```

---

## 5. Evaluating an Existing Checkpoint Without Retraining

To evaluate an already-trained model checkpoint directly on 20 held-out test scenarios:

```powershell
.\.venv\Scripts\python.exe run_all.py --skip-train --checkpoint ".\runs\run_20261014_192552\checkpoints\best_model.zip" --eval-episodes 20
```

---

## 6. (Optional) Launching the Interactive Streamlit Dashboard

If you wish to inspect results interactively in your browser:
```powershell
# Install dashboard dependency if not already installed
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt

# Start Streamlit dashboard
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```
Open `http://localhost:8501` in your browser. Press `Ctrl + C` in PowerShell to stop the dashboard server when done.
