"""Single source of truth for chosen assignment values."""
from pathlib import Path

STUDENT_NAME = "Rajesh Garg"
STUDENT_ID = "2026AJ05869"

# Clarification: choose our own values and state them in README/report.
HIDDEN_SIZE = 64          # Same h for RNN, LSTM and GRU.
D_MODEL = 128             # Transformer model width d.
N_HEADS = 4               # D_MODEL must be divisible by N_HEADS.
SEEDS = [7, 17, 27]       # Any three seeds are allowed.

SMOKE_MODE = True         # Change to False before collecting final evidence.
DEVICE_PREFERENCE = "cuda"
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
HAR_EPOCHS = 2 if SMOKE_MODE else 30
ADDING_EPOCHS = 1 if SMOKE_MODE else 40

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_CSV = ROOT / "results.csv"
P1_FIGURES = ROOT / "problem1" / "figures"
P2_FIGURES = ROOT / "problem2" / "figures"

def validate_choices():
    assert STUDENT_ID == "2026AJ05869"
    assert HIDDEN_SIZE > 0 and D_MODEL > 0
    assert D_MODEL % N_HEADS == 0, "d must be divisible by head count"
    assert len(SEEDS) == 3 and len(set(SEEDS)) == 3

def assigned_ablation():
    """Separate student-ID ablation retained from the original task."""
    return "lstm_forget_bias"  # ID ends in odd digit 9.
