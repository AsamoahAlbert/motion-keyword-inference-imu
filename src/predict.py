import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MODEL_PATH = "best_cnn_acc_only.pt"

TOP10 = [
    "california", "closed", "continue", "expect", "forecast",
    "predicted", "remind", "reminded", "tomorrow", "united"
]

SEQ_LEN = 256


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


def read_acc_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    stealth_cols = ["acc_x", "acc_y", "acc_z"]
    pi_cols = ["ax_g", "ay_g", "az_g"]

    if all(c in df.columns for c in stealth_cols):
        arr = df[stealth_cols].to_numpy(dtype=np.float32)
        if len(arr) > 0:
            return arr

    if all(c in df.columns for c in pi_cols):
        arr = df[pi_cols].to_numpy(dtype=np.float32)
        if len(arr) > 0:
            return arr

    raise ValueError(f"Could not parse accelerometer columns from file: {path}")


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


def load_one_sample(acc_path: str, seq_len: int) -> torch.Tensor:
    acc = read_acc_csv(acc_path)
    acc = resize_sequence(acc, seq_len)   # [T, 3]
    x = acc.T.astype(np.float32)          # [3, T]
    x = normalize_per_channel(x)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 3, T]


def main():
    parser = argparse.ArgumentParser(description="Inference for acc-only CNN keyword model")
    parser.add_argument("--acc_path", required=True, help="Path to accelerometer CSV")
    parser.add_argument("--model_path", default=MODEL_PATH, help="Path to trained acc-only model")
    args = parser.parse_args()

    acc_path = Path(args.acc_path)
    model_path = Path(args.model_path)

    if not acc_path.exists():
        raise FileNotFoundError(f"ACC file not found: {acc_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    labels_sorted = sorted(TOP10)
    idx_to_label = {i: lbl for i, lbl in enumerate(labels_sorted)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ACC file: {acc_path}")
    print(f"Model: {model_path}")

    model = ACC_CNN(num_classes=len(labels_sorted), dropout=0.4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = load_one_sample(str(acc_path), SEQ_LEN).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_label[pred_idx]

    print("\nPredicted label:")
    print(f"  {pred_label}")

    print("\nTop 5 probabilities:")
    ranking = sorted(
        [(idx_to_label[i], float(probs[i])) for i in range(len(probs))],
        key=lambda z: z[1],
        reverse=True
    )
    for label, prob in ranking[:5]:
        print(f"  {label:12s}  {prob:.4f}")


if __name__ == "__main__":
    main()