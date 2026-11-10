"""Dataset acquisition and preprocessing using relative paths only."""
from pathlib import Path
import urllib.request, zipfile
import numpy as np
import torch
from torch.utils.data import TensorDataset

HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def make_adding(n: int, T: int, seed: int):
    """Generate the exact two-channel long-memory task specified in the brief."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    x[np.arange(n), i1, 1] = 1
    x[np.arange(n), i2, 1] = 1
    y = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    return x, y.astype(np.float32)

def assert_adding_data_is_valid():
    """Required data test: two markers, one per half, and correct target."""
    x, y = make_adding(256, 100, 123)
    assert np.all(x[:, :, 1].sum(axis=1) == 2)
    assert np.all(x[:, :50, 1].sum(axis=1) == 1)
    assert np.all(x[:, 50:, 1].sum(axis=1) == 1)
    assert np.allclose((x[:, :, 0] * x[:, :, 1]).sum(axis=1), y)

def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target

def download_har(data_dir: Path) -> Path:
    """Download the outer archive and also extract the nested UCI ZIP when present."""
    data_dir = Path(data_dir)
    archive = _download(HAR_URL, data_dir / "har.zip")
    extract_dir = data_dir / "har"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not list(extract_dir.rglob("UCI HAR Dataset")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        nested = list(extract_dir.rglob("UCI HAR Dataset.zip"))
        if nested:
            with zipfile.ZipFile(nested[0]) as zf:
                zf.extractall(extract_dir)
    matches = list(extract_dir.rglob("UCI HAR Dataset"))
    if not matches:
        raise FileNotFoundError("UCI HAR Dataset folder was not found after extraction.")
    return matches[0]

def _load_har_split(root: Path, split: str):
    signal_dir = root / split / "Inertial Signals"
    arrays = [np.loadtxt(signal_dir / f"{c}_{split}.txt", dtype=np.float32) for c in CHANNELS]
    x = np.stack(arrays, axis=-1)  # [samples, 128 time steps, 9 channels]
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return x, y, subjects

def subjectwise_har(seed: int, data_dir: Path):
    """Hold out four training subjects and normalize with training-only statistics."""
    root = download_har(data_dir)
    x, y, subjects = _load_har_split(root, "train")
    xt, yt, test_subjects = _load_har_split(root, "test")
    val_subjects = np.random.default_rng(seed).choice(np.unique(subjects), 4, replace=False)
    val_mask = np.isin(subjects, val_subjects)
    train_mask = ~val_mask
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = x[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    normalize = lambda a: (a - mean) / std
    to_ds = lambda a, b: TensorDataset(torch.tensor(a), torch.tensor(b))
    return (
        to_ds(normalize(x[train_mask]), y[train_mask]),
        to_ds(normalize(x[val_mask]), y[val_mask]),
        to_ds(normalize(xt), yt),
        {"mean": mean, "std": std, "val_subjects": val_subjects, "test_subjects": test_subjects},
    )

def download_shakespeare(data_dir: Path) -> str:
    path = _download(SHAKESPEARE_URL, Path(data_dir) / "tiny_shakespeare.txt")
    return path.read_text(encoding="utf-8")
