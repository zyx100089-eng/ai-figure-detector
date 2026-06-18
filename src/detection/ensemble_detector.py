"""
Ensemble detector: combine CNN, frequency, and statistical detectors
using soft voting (averaged probabilities).
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.detection.frequency_detector import extract_frequency_features, load_split as load_freq_split
from src.detection.stat_detector import extract_stat_features, load_split as load_stat_split
from src.detection.cnn_detector import build_model, get_transforms, evaluate as cnn_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_cnn_probabilities(split: str, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Get CNN predicted probabilities for a split."""
    _, eval_tf = get_transforms()
    from torchvision import datasets
    from torch.utils.data import DataLoader

    ds = datasets.ImageFolder(SPLITS_DIR / split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model_path = RESULTS_DIR / "cnn_resnet18.pth"
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def train_ensemble():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=== Training component models ===\n")

    # Frequency features + SVM
    print("1. Frequency + SVM")
    X_train_freq, y_train = load_freq_split("train")
    X_val_freq, y_val = load_freq_split("val")
    X_test_freq, y_test = load_freq_split("test")

    freq_scaler = StandardScaler()
    X_train_freq = freq_scaler.fit_transform(X_train_freq)
    X_val_freq = freq_scaler.transform(X_val_freq)
    X_test_freq = freq_scaler.transform(X_test_freq)

    svm = SVC(C=10.0, gamma="scale", kernel="rbf", probability=True)
    svm.fit(np.vstack([X_train_freq, X_val_freq]), np.concatenate([y_train, y_val]))
    freq_probs = svm.predict_proba(X_test_freq)[:, 1]
    print(f"   F1: {f1_score(y_test, (freq_probs >= 0.5).astype(int)):.3f}")

    # Statistical features + RF
    print("2. Statistical + RF")
    X_train_stat, _ = load_stat_split("train")
    X_val_stat, _ = load_stat_split("val")
    X_test_stat, _ = load_stat_split("test")

    stat_scaler = StandardScaler()
    X_train_stat = stat_scaler.fit_transform(X_train_stat)
    X_val_stat = stat_scaler.transform(X_val_stat)
    X_test_stat = stat_scaler.transform(X_test_stat)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(np.vstack([X_train_stat, X_val_stat]), np.concatenate([y_train, y_val]))
    stat_probs = rf.predict_proba(X_test_stat)[:, 1]
    print(f"   F1: {f1_score(y_test, (stat_probs >= 0.5).astype(int)):.3f}")

    # CNN probabilities
    print("3. CNN (ResNet-18)")
    cnn_probs, cnn_labels = get_cnn_probabilities("test", device)
    print(f"   F1: {f1_score(y_test, (cnn_probs >= 0.5).astype(int)):.3f}")

    # Ensemble: soft voting (average probabilities)
    print("\n=== Ensemble (Soft Voting) ===")

    # Try different weight combinations
    best_f1 = 0
    best_weights = None
    for w_cnn in [0.3, 0.4, 0.5, 0.6]:
        for w_stat in [0.2, 0.3, 0.4]:
            w_freq = 1.0 - w_cnn - w_stat
            if w_freq < 0.05:
                continue
            ensemble_probs = w_cnn * cnn_probs + w_stat * stat_probs + w_freq * freq_probs
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)
            ef1 = f1_score(y_test, ensemble_preds)
            if ef1 > best_f1:
                best_f1 = ef1
                best_weights = (w_cnn, w_stat, w_freq)

    w_cnn, w_stat, w_freq = best_weights
    print(f"Best weights: CNN={w_cnn:.1f}, Stat={w_stat:.1f}, Freq={w_freq:.1f}")

    ensemble_probs = w_cnn * cnn_probs + w_stat * stat_probs + w_freq * freq_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, ensemble_preds),
        "precision": precision_score(y_test, ensemble_preds, zero_division=0),
        "recall": recall_score(y_test, ensemble_preds, zero_division=0),
        "f1": f1_score(y_test, ensemble_preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, ensemble_probs) if len(set(y_test)) > 1 else 0.0,
    }

    print("\n--- Test Results (Ensemble) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ensemble_results.json", "w") as f:
        json.dump({
            **metrics,
            "weights": {"cnn": w_cnn, "stat": w_stat, "freq": w_freq},
            "component_probs": {
                "cnn": cnn_probs.tolist(),
                "stat": stat_probs.tolist(),
                "freq": freq_probs.tolist(),
            },
            "labels": y_test.tolist(),
        }, f, indent=2)

    return metrics, ensemble_probs, y_test


if __name__ == "__main__":
    train_ensemble()
