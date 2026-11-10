"""Central configuration. Change values here instead of editing many notebook cells."""
from pathlib import Path

STUDENT_ID = "2026AJ05869"
SEEDS = [7, 17, 27]          # Keeps exactly three reproducible seeds.
SMOKE_MODE = True            # Sets False for final experiments.
DEVICE_PREFERENCE = "cuda"  # Falls back to CPU automatically.

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_CSV = ROOT / "results.csv"
P1_FIGURES = ROOT / "problem1" / "figures"
P2_FIGURES = ROOT / "problem2" / "figures"

# Starting hyperparameters. Keep budgets equal within each comparison.
BATCH_SIZE = 128
HAR_HIDDEN_SIZE = 64
HAR_EPOCHS = 2 if SMOKE_MODE else 30
ADDING_EPOCHS = 1 if SMOKE_MODE else 40
LEARNING_RATE = 1e-3

def last_student_digit():
    digits = [c for c in STUDENT_ID if c.isdigit()]
    if not digits:
        raise ValueError("Set STUDENT_ID in config.py before final runs.")
    return int(digits[-1])

def assigned_ablation():
    """Even ID: RNN clipping. Odd ID: LSTM forget-bias experiment."""
    return "rnn_gradient_clipping" if last_student_digit() % 2 == 0 else "lstm_forget_bias"
