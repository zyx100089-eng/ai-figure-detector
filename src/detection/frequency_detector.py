"""
Frequency-domain detector: extract DCT/FFT spectral features and train
an SVM to distinguish real from AI-generated scientific figures.

Rationale: AI-generated images often have different high-frequency patterns
compared to real screenshots/renders of matplotlib charts.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_frequency_features(image_path: str, size: int = 256) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(48)
    img = cv2.resize(img, (size, size))
    img_float = img.astype(np.float32) / 255.0

    # 2D DCT
    dct = cv2.dct(img_float)
    dct_log = np.log1p(np.abs(dct))

    # Divide into frequency bands (low, mid, high)
    h, w = dct_log.shape
    bands = [
        dct_log[:h//4, :w//4],         # low frequency
        dct_log[:h//2, :w//2],         # low+mid frequency
        dct_log[h//4:3*h//4, w//4:3*w//4],  # mid frequency
        dct_log[h//2:, w//2:],         # high frequency
    ]

    features = []
    for band in bands:
        features.extend([
            band.mean(),
            band.std(),
            np.median(band),
            np.percentile(band, 90),
        ])

    # 2D FFT
    fft = np.fft.fft2(img_float)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    # Radial frequency profile
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    radial_bins = 8
    bin_size = max_r // radial_bins
    for i in range(radial_bins):
        r_inner = i * bin_size
        r_outer = (i + 1) * bin_size
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x**2 + y**2 >= r_inner**2) & (x**2 + y**2 < r_outer**2)
        ring = magnitude[mask]
        features.append(ring.mean() if len(ring) > 0 else 0)
        features.append(ring.std() if len(ring) > 0 else 0)

    return np.array(features)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = SPLITS_DIR / split
    features_list = []
    labels = []

    for label_name in ["real", "fake"]:
        label_dir = split_dir / label_name
        label_val = 0 if label_name == "fake" else 1
        images = sorted(label_dir.glob("*.png"))
        for img_path in tqdm(images, desc=f"{split}/{label_name}"):
            feat = extract_frequency_features(str(img_path))
            features_list.append(feat)
            labels.append(label_val)

    return np.array(features_list), np.array(labels)


def train_frequency_detector():
    print("Extracting frequency features...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nFeature dim: {X_train.shape[1]}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    best_score = 0
    best_params = {}
    for C in [0.1, 1.0, 10.0]:
        for gamma in ["scale", "auto"]:
            svm = SVC(C=C, gamma=gamma, kernel="rbf", probability=True)
            svm.fit(X_train, y_train)
            val_pred = svm.predict(X_val)
            val_f1 = f1_score(y_val, val_pred)
            if val_f1 > best_score:
                best_score = val_f1
                best_params = {"C": C, "gamma": gamma}

    print(f"Best params: {best_params} (val F1: {best_score:.3f})")

    svm = SVC(**best_params, kernel="rbf", probability=True)
    svm.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0,
    }

    print("\n--- Test Results (Frequency + SVM) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "frequency_svm_results.json", "w") as f:
        json.dump({**metrics, "best_params": best_params}, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_frequency_detector()
