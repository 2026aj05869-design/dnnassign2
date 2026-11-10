"""Small, reusable loops. Experiment grids remain visible in the notebooks."""
import time
import torch
from sklearn.metrics import accuracy_score, f1_score


def train_epoch(model, loader, optimizer, criterion, device, clip_norm=None):
    model.train()
    total_loss = 0.0
    start = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(x)
    elapsed = time.perf_counter() - start
    return total_loss / len(loader.dataset), elapsed

@torch.no_grad()
def classification_metrics(model, loader, device):
    model.eval()
    truth, predicted = [], []
    for x, y in loader:
        logits = model(x.to(device))
        truth.extend(y.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return (
        accuracy_score(truth, predicted),
        f1_score(truth, predicted, average="macro"),
        truth,
        predicted,
    )

@torch.no_grad()
def regression_mse(model, loader, device):
    model.eval()
    squared_error, count = 0.0, 0
    for x, y in loader:
        p = model(x.to(device)).squeeze(-1).cpu()
        squared_error += ((p - y) ** 2).sum().item()
        count += len(y)
    return squared_error / count
