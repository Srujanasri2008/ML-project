# 🚀 Complete VS Code Execution Guide

**Project:** Soft Actor-Critic-Based Semantic-Aware Power and Time Allocation for Hybrid NOMA–OFDMA Networks  
**Location:** `C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma`

This beginner-friendly guide walks you step-by-step through opening, testing, running, and inspecting this project inside **Visual Studio Code (VS Code)** on Windows.

---

## 📌 Table of Contents
1. [Opening the Project in VS Code](#1-opening-the-project-in-vs-code)
2. [Configuring the Python Interpreter in VS Code](#2-configuring-the-python-interpreter-in-vs-code)
3. [Opening the Integrated Terminal](#3-opening-the-integrated-terminal)
4. [Execution Commands (Copy & Paste)](#4-execution-commands-copy--paste)
   - [Command 1: Run Preflight System Check](#command-1-run-preflight-system-check)
   - [Command 2: Run All 25 Automated Unit Tests](#command-2-run-all-25-automated-unit-tests)
   - [Command 3: Run Fast Smoke Training (1,000 Steps)](#command-3-run-fast-smoke-training-1000-steps)
   - [Command 4: Run Standard Training & Evaluation (20,000 Steps)](#command-4-run-standard-training--evaluation-20000-steps)
   - [Command 5: Evaluate Trained Checkpoint Without Retraining](#command-5-evaluate-trained-checkpoint-without-retraining)
   - [Command 6: Launch Interactive Streamlit Dashboard](#command-6-launch-interactive-streamlit-dashboard)
5. [Viewing Results, CSV Tables & Plots Inside VS Code](#5-viewing-results-csv-tables--plots-inside-vs-code)
6. [Troubleshooting & Helpful Tips](#6-troubleshooting--helpful-tips)

---

## 1. Opening the Project in VS Code

1. Launch **Visual Studio Code**.
2. Click on the top menu: **`File`** ➔ **`Open Folder...`** (or press `Ctrl + K, Ctrl + O`).
3. Paste or navigate to this folder path:
   ```text
   C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma
   ```
4. Click **Select Folder**.
5. When prompted *"Do you trust the authors of the files in this folder?"*, click **Yes, I trust the authors**.

---

## 2. Configuring the Python Interpreter in VS Code

To make sure VS Code uses the dedicated project virtual environment (`.venv`) with PyTorch CPU and Stable-Baselines3:

1. Press `Ctrl + Shift + P` to open the VS Code Command Palette.
2. Type: `Python: Select Interpreter` and press **Enter**.
3. Choose the interpreter inside `.venv`:
   ```text
   Python 3.12.x ('.venv': venv)  .\.venv\Scripts\python.exe
   ```
4. If you don't see it listed immediately, click **`Enter interpreter path...`** ➔ **`Find...`** and select:
   `C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma\.venv\Scripts\python.exe`.

---

## 3. Opening the Integrated Terminal

1. In VS Code, open the integrated terminal by:
   - Pressing `Ctrl + ` ` (the backtick key, next to 1 on your keyboard), OR
   - Clicking on the top menu: **`Terminal`** ➔ **`New Terminal`**.
2. Make sure the prompt shows you are in the project folder:
   ```powershell
   PS C:\Users\SRUJANA BANDARU\.gemini\antigravity\scratch\semantic_noma_ofdma>
   ```

---

## 4. Execution Commands (Copy & Paste)

You can copy and run any of the commands below directly in the VS Code terminal.

### Command 1: Run Preflight System Check
Verifies your Python environment, RAM, and free disk space ($> 5$ GB):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```
*Expected output: Disk space met, virtual environment Python 3.12 detected, PyTorch installed.*

---

### Command 2: Run All 25 Automated Unit Tests
Verifies mathematical equations, power constraints, airtime feasibility, Gymnasium compliance, and checkpoint loading:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
*Expected output: `25 passed in ~6.5s` (100% pass rate).*

---

### Command 3: Run Fast Smoke Training (1,000 Steps)
Runs a quick 1,000-timestep test to verify the complete training, checkpointing, and evaluation pipeline:
```powershell
.\.venv\Scripts\python.exe run_all.py --smoke
```
*Expected output: Completes in ~18 seconds, saves checkpoint and plots into a new directory under `runs/`.*

---

### Command 4: Run Standard Training & Evaluation (20,000 Steps)
Executes the full research experiment:
- Trains the Soft Actor-Critic agent for 20,000 timesteps using maximum 2 CPU threads.
- Validates the model every 2,000 steps across 5 validation seeds.
- Evaluates the best checkpoint against **3 baselines** across **20 held-out test scenarios** (seeds 2000–2019).
```powershell
.\.venv\Scripts\python.exe run_all.py --timesteps 20000 --max-runtime-seconds 600 --eval-episodes 20
```
*Expected output: Completes in ~95 seconds, prints comparison table, and generates all 9 publication plots.*

---

### Command 5: Evaluate Trained Checkpoint Without Retraining
If you already have a trained model checkpoint and want to evaluate it directly on test scenarios without waiting for training:
```powershell
.\.venv\Scripts\python.exe run_all.py --skip-train --checkpoint ".\runs\run_20261014_192552\checkpoints\best_model.zip" --eval-episodes 20
```
*Expected output: Evaluates all 4 policies on 20 test seeds in ~8 seconds and updates evaluation tables.*

---

### Command 6: Launch Interactive Streamlit Dashboard
Launches a browser dashboard where you can interactively explore training curves, results tables, and allocation plots:
```powershell
# 1. Install dashboard requirement (one-time)
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt

# 2. Start the dashboard
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```
*Expected output: Streamlit opens in your browser at `http://localhost:8501`. Press `Ctrl + C` in the terminal to stop the server when done.*

---

## 5. Viewing Results, CSV Tables & Plots Inside VS Code

All generated outputs from the standard run are located in:  
📁 `runs\run_20261014_192552\`

In the VS Code File Explorer (left sidebar), click to open:

### 1. View Summary Tables Directly in VS Code
- **`evaluation_summary.csv`**: Contains Mean $\pm$ Standard Deviation for all policies.
- **`per_seed_metrics.csv`**: Contains episode-by-episode raw numbers for each of the 20 test seeds.
- **`training_log.csv`**: Contains step-by-step rewards and penalty progression.

> *Tip for viewing CSVs in VS Code:* You can install the free VS Code extension **"Rainbow CSV"** or **"Data Wrangler"** for formatted table views. Alternatively, run this in the terminal:
> ```powershell
> Get-Content ".\runs\run_20261014_192552\evaluation_summary.csv" | ConvertFrom-Csv | Format-Table -AutoSize
> ```

### 2. View Plot Images in VS Code
Simply click on any `.png` file in the `runs\run_20261014_192552\` folder inside VS Code's sidebar to view the high-resolution graph:
- 📊 **`training_reward.png`**: SAC training learning curve.
- 📉 **`reward_components.png`**: Semantic utility vs energy and deadline penalties.
- 📈 **`semantic_utility_comparison.png`**: Bar chart comparing SAC, Equal, Semantic Priority, and Random.
- ⚡ **`energy_efficiency_comparison.png`**: Bits per Joule comparison (SAC achieves $+79.8\%$ higher efficiency).
- 📶 **`throughput_comparison.png`**: Goodput vs achievable sum-rate.
- ⏱️ **`latency_comparison.png`**: Task latency vs deadline miss percentage.
- ⚖️ **`fairness_comparison.png`**: Jain's fairness index across users.
- 🔋 **`power_allocation_example.png`**: Dynamic power allocation across pairs.
- 🕒 **`airtime_allocation_example.png`**: Dynamic airtime sharing across pairs.

---

## 6. Troubleshooting & Helpful Tips

### Q1: VS Code says "File cannot be loaded because running scripts is disabled on this system"?
**Solution:** PowerShell execution policy restricts scripts by default. Run this in your terminal once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Or always call Python via `.\.venv\Scripts\python.exe` directly (which bypasses script execution policies).

### Q2: Do I need to reinstall dependencies every time I open VS Code?
**No!** The virtual environment `.venv` is persistent. As long as you run commands using `.\.venv\Scripts\python.exe`, all packages (PyTorch, Stable-Baselines3, Gymnasium, etc.) remain permanently installed and ready.

### Q3: How do I change simulation parameters (e.g., number of users, power, deadlines)?
All parameters are cleanly configured in:
- [`configs/default.json`](file:///C:/Users/SRUJANA%20BANDARU/.gemini/antigravity/scratch/semantic_noma_ofdma/configs/default.json)  
You can edit values (e.g., change `p_max_w`, `total_timesteps`, or `importance_range`) directly in VS Code and rerun `run_all.py`.
