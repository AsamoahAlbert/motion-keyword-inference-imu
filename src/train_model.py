"""
train_model.py

Trains a 1D CNN model on IMU vibration data for keyword classification.

Steps:
- Load dataset
- Preprocess signals
- Train model
- Evaluate performance
"""

import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# =========================================================
# Configuration
# =========================================================
DATASET_CSV = r"./metadata/keyword_labeled_samples_imu_only.csv"

TOP10 = [
    "california", "closed", "continue", "expect", "forecast",
    "predicted", "remind", "reminded", "tomorrow", "united"
]

SEQ_LEN = 256
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
MIN_SAMPLES_PER_CLASS = 2

USE_ENV_SPLIT = False
NEW_ENVS = ["cleanair", "cleantime", "cleansun", "cleanstock"]
ENV_COL = "env"


# =========================================================
# Timer
# =========================================================
class Timer:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.total_start = None
        self.total_time = None

    def start_total(self):
        self.total_start = time.time()

    def stop_total(self):
        if self.total_start is not None:
            self.total_time = time.time() - self.total_start

    def start(self, name: str):
        self.start_times[name] = time.time()

    def stop(self, name: str):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name in self.timings:
                self.timings[name]["total"] += elapsed
                self.timings[name]["count"] += 1
            else:
                self.timings[name] = {"total": elapsed, "count": 1}
            del self.start_times[name]
            return elapsed
        return 0

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

    def summary(self) -> str:
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("TIMING REPORT")
        lines.append("=" * 60)

        for name, data in self.timings.items():
            total = data["total"]
            count = data["count"]
            avg = total / count if count > 0 else 0

            if count > 1:
                lines.append(f"{name}:")
                lines.append(f"  Total: {self.format_time(total)}")
                lines.append(f"  Count: {count}")
                lines.append(f"  Average: {self.format_time(avg)}")
            else:
                lines.append(f"{name}: {self.format_time(total)}")

        if self.total_time is not None:
            lines.append("-" * 60)
            lines.append(f"TOTAL TIME: {self.format_time(self.total_time)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, data in self.timings.items():
            rows.append({
                "phase": name,
                "total_seconds": round(data["total"], 3),
                "count": data["count"],
                "avg_seconds": round(data["total"] / data["count"], 3) if data["count"] > 0 else 0
            })
        if self.total_time is not None:
            rows.append({
                "phase": "TOTAL",
                "total_seconds": round(self.total_time, 3),
                "count": 1,
                "avg_seconds": round(self.total_time, 3)
            })
        return pd.DataFrame(rows)


# =========================================================
# Utility functions
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_env_from_path(path_str: str) -> str:
    p = Path(str(path_str))
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def read_acc_csv(path: str) -> np.ndarray | None:
    cols = ["acc_x", "acc_y", "acc_z"]

    try:
        df = pd.read_csv(path)
        if all(c in df.columns for c in cols):
            arr = df[cols].to_numpy(dtype=np.float32)
            if len(arr) > 0:
                return arr
    except Exception:
        pass

    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] >= 4:
            arr = df.iloc[:, 1:4].to_numpy(dtype=np.float32)
            if len(arr) > 0:
                return arr
        elif df.shape[1] >= 3:
            arr = df.iloc[:, 0:3].to_numpy(dtype=np.float32)
            if len(arr) > 0:
                return arr
    except Exception:
        pass

    return None


def resize_sequence(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return np.zeros((target_len, 3), dtype=np.float32)

    n = len(arr)
    if n == target_len:
        return arr.astype(np.float32)

    old_idx = np.linspace(0, 1, n, dtype=np.float32)
    new_idx = np.linspace(0, 1, target_len, dtype=np.float32)

    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(new_idx, old_idx, arr[:, j]).astype(np.float32)
    return out


def normalize_per_channel(x: np.ndarray) -> np.ndarray:
    eps = 1e-6
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


# =========================================================
# Dataset
# =========================================================
class AccDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, seq_len=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_acc(self, acc_path: str) -> torch.Tensor:
        acc = read_acc_csv(acc_path)

        if acc is None:
            x = np.zeros((3, self.seq_len), dtype=np.float32)
            return torch.from_numpy(x)

        acc = resize_sequence(acc, self.seq_len)  # [T, 3]
        x = acc.T.astype(np.float32)              # [3, T]

        if self.augment:
            gain = np.random.uniform(0.9, 1.1)
            x = x * gain

            if np.random.rand() < 0.4:
                noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
                x = x + noise

            if np.random.rand() < 0.4:
                t0 = np.random.randint(0, self.seq_len)
                tlen = np.random.randint(5, max(6, self.seq_len // 10))
                x[:, t0:min(self.seq_len, t0 + tlen)] = 0.0

        x = normalize_per_channel(x)
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        acc_path = self.df.loc[idx, "acc_path"]
        label = self.df.loc[idx, "label"]

        x = self._load_acc(acc_path)
        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return x, y


# =========================================================
# Model
# =========================================================
class ACC_CNN(nn.Module):
    def __init__(self, num_classes: int, dropout=0.4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device, idx_to_label):
    model.eval()
    all_logits, all_y = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    y_pred = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    rows = []
    for j in sorted(np.unique(y_true)):
        name = idx_to_label[j]
        mask_true = (y_true == j).astype(int)
        mask_pred = (y_pred == j).astype(int)
        p = precision_score(mask_true, mask_pred, zero_division=0)
        r = recall_score(mask_true, mask_pred, zero_division=0)
        f = f1_score(mask_true, mask_pred, zero_division=0)
        support = int(mask_true.sum())
        rows.append((name, support, p, r, f))

    out = pd.DataFrame(rows, columns=["label", "support", "precision", "recall", "f1"])
    out = out.sort_values("f1", ascending=False)

    return acc, macro_f1, weighted_f1, out, y_true, y_pred


# =========================================================
# Main
# =========================================================
def main():
    timer = Timer()
    timer.start_total()
    set_seed(SEED)

    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------
    # Data loading
    # -----------------------------------------------------
    timer.start("Data loading")

    df = pd.read_csv(DATASET_CSV)

    required = {"label", "acc_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {required}, Actual: {list(df.columns)}")

    if ENV_COL not in df.columns:
        df[ENV_COL] = df["acc_path"].apply(extract_env_from_path)

    df = df[df["label"].isin(TOP10)].copy()

    counts = df["label"].value_counts()
    valid_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    dropped_labels = counts[counts < MIN_SAMPLES_PER_CLASS]

    if len(dropped_labels) > 0:
        print("\nDropped labels with too few samples:")
        for lbl, cnt in dropped_labels.items():
            print(f"  {lbl:15s} {cnt}")

    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    print(f"\nValid samples after filtering: {len(df)}")
    print("\nSamples per label:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"  {lbl:15s} {cnt}")

    labels_sorted = sorted(df["label"].unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------
    if USE_ENV_SPLIT:
        is_new = df[ENV_COL].astype(str).isin(NEW_ENVS)
        df_test = df[is_new].reset_index(drop=True)
        df_old = df[~is_new].reset_index(drop=True)

        if len(df_test) == 0:
            raise ValueError(f"NEW_ENVS={NEW_ENVS} did not match any rows")

        df_train, df_val = train_test_split(
            df_old,
            test_size=0.20,
            random_state=SEED,
            stratify=df_old["label"],
        )

        print(f"\nData split (by environment):")
        print(f"  train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    else:
        df_train, df_test = train_test_split(
            df,
            test_size=0.30,
            random_state=SEED,
            stratify=df["label"],
        )
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.20,
            random_state=SEED,
            stratify=df_train["label"],
        )
        print(f"\nData split (random):")
        print(f"  train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    train_ds = AccDataset(df_train, label_to_idx, seq_len=SEQ_LEN, augment=True)
    val_ds = AccDataset(df_val, label_to_idx, seq_len=SEQ_LEN, augment=False)
    test_ds = AccDataset(df_test, label_to_idx, seq_len=SEQ_LEN, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    timer.stop("Data loading")

    # -----------------------------------------------------
    # Model initialization
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    timer.start("Model initialization")

    model = ACC_CNN(num_classes=len(labels_sorted), dropout=0.4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    y_train_labels = df_train["label"].to_numpy()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(labels_sorted),
        y=y_train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nClass weights:")
    for label, weight in zip(labels_sorted, class_weights.cpu().numpy()):
        print(f"  {label:12s} -> {weight:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    timer.stop("Model initialization")

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    best_val_macro = -1.0
    best_path = "best_cnn_acc_only.pt"
    patience = 5
    no_improve = 0

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_macro_history = []
    lr_history = []

    print("\nStarting training...")
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        timer.start("Training (per epoch)")
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        current_lr = scheduler.get_last_lr()[0]
        timer.stop("Training (per epoch)")

        # Real per-epoch training accuracy
        timer.start("Train evaluation (per epoch)")
        train_acc, train_macro, train_weighted, _, _, _ = evaluate(model, train_loader, device, idx_to_label)
        timer.stop("Train evaluation (per epoch)")

        timer.start("Validation (per epoch)")
        val_acc, val_macro, val_weighted, val_table, _, _ = evaluate(model, val_loader, device, idx_to_label)
        timer.stop("Validation (per epoch)")

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(avg_loss)
        val_macro_history.append(val_macro)
        lr_history.append(current_lr)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"\nEpoch {epoch}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"train acc={train_acc:.4f}  "
            f"val acc={val_acc:.4f}  "
            f"val macro-F1={val_macro:.4f}  "
            f"lr={current_lr:.6f}  "
            f"time={epoch_time:.1f}s"
        )
        print("Val Top-5:")
        print(val_table.head(5).to_string(index=False))

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (macro-F1={best_val_macro:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, early stopping")
                break

    actual_epochs = len(epoch_times)
    avg_epoch_time = sum(epoch_times) / actual_epochs if epoch_times else 0
    total_training_time = sum(epoch_times)

    print(f"\nTraining completed:")
    print(f"  Actual epochs: {actual_epochs}")
    print(f"  Total training time: {timer.format_time(total_training_time)}")
    print(f"  Average time per epoch: {avg_epoch_time:.1f}s")

    # Save training history
    history_df = pd.DataFrame({
        "epoch": list(range(1, len(train_acc_history) + 1)),
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_loss": train_loss_history,
        "val_macro_f1": val_macro_history,
        "lr": lr_history,
    })
    history_df.to_csv("training_history.csv", index=False, encoding="utf-8")
    print("Saved training_history.csv")

    # -----------------------------------------------------
    # Testing
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading best model for testing...")

    timer.start("Model loading")
    model.load_state_dict(torch.load(best_path, map_location=device))
    timer.stop("Model loading")

    timer.start("Test evaluation")
    test_acc, test_macro, test_weighted, test_table, y_true, y_pred = evaluate(model, test_loader, device, idx_to_label)
    timer.stop("Test evaluation")

    print("\nTest results:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Macro-F1:    {test_macro:.4f}")
    print(f"  Weighted-F1: {test_weighted:.4f}")
    print("\nPer-label details:")
    print(test_table.to_string(index=False))

    test_table.to_csv("cnn_acc_only_report.csv", index=False, encoding="utf-8")

    # Save predictions
    timer.start("Saving results")
    rows = []
    df_test_reset = df_test.reset_index(drop=True)
    for i in range(len(df_test_reset)):
        rows.append({
            "acc_path": df_test_reset.iloc[i]["acc_path"],
            "label_true": idx_to_label[int(y_true[i])],
            "label_pred": idx_to_label[int(y_pred[i])],
            "env": df_test_reset.iloc[i][ENV_COL] if ENV_COL in df_test_reset.columns else "unknown",
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv("cnn_acc_only_predictions.csv", index=False, encoding="utf-8")
    timer.stop("Saving results")

    timer.stop_total()

    print(timer.summary())
    timer.to_dataframe().to_csv("cnn_acc_only_timing.csv", index=False, encoding="utf-8")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSaved files:")
    print("  - best_cnn_acc_only.pt")
    print("  - training_history.csv")
    print("  - cnn_acc_only_report.csv")
    print("  - cnn_acc_only_predictions.csv")
    print("  - cnn_acc_only_timing.csv")


if __name__ == "__main__":
    main()
