"""Reproducibility, device selection, parameter counting and result logging."""
from pathlib import Path
import csv, math, random
import numpy as np
import torch

RESULT_COLUMNS = [
    "problem", "model", "seed", "seq_len", "test_acc", "macro_f1",
    "test_mse", "params", "sec_per_epoch", "val_loss_nats",
    "bits_per_char", "notes"
]

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. GPU results may still differ slightly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(preference: str = "cuda") -> torch.device:
    return torch.device("cuda" if preference == "cuda" and torch.cuda.is_available() else "cpu")

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bits_per_character(loss_nats: float) -> float:
    return float(loss_nats) / math.log(2.0)

def append_result(path, **values) -> None:
    """Append one experiment immediately, preserving the exact required schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {name: values.get(name, "") for name in RESULT_COLUMNS}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
