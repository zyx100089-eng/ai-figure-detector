"""
Statistical consistency detector: extract visual/structural features from
scientific figures and use a Random Forest to detect AI-generated ones.

Features capture patterns that AI often gets wrong:
- Color distribution and palette complexity
- Edge density and sharpness
- Text region statistics (via simple thresholding)
- Symmetry and layout regularity
- Noise patterns
"""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_stat_features(image_path: str, size: int = 256) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(60)
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []

    # Color statistics per channel (BGR + HSV)
    for ch in range(3):
        channel = img[:, :, ch].astype(float)
        features.extend([channel.mean(), channel.std(), np.median(channel)])
    for ch in range(3):
        channel = hsv[:, :, ch].astype(float)
        features.extend([channel.mean(), channel.std()])

    # Color diversity: unique colors in downsampled image
    small = cv2.resize(img, (64, 64))
    unique_colors = len(np.unique(small.reshape(-1, 3), axis=0))
    features.append(unique_colors / (64 * 64))

    # Edge features
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    features.append(edge_density)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([laplacian.mean(), laplacian.std(), np.abs(laplacian).mean()])

    # White/background ratio
    white_mask = gray > 240
    features.append(white_mask.mean())

    # Text-like region detection (thin dark strokes on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_ratio = binary.mean() / 255.0
    features.append(text_ratio)

    # Line detection (Hough)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = len(lines) if lines is not None else 0
    features.append(num_lines)

    if lines is not None and len(lines) > 0:
        angles = []
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angles.append(angle)
            lengths.append(length)
        features.extend([np.mean(lengths), np.std(lengths)])
        h_lines = sum(1 for a in angles if abs(a) < 10 or abs(a) > 170)
        v_lines = sum(1 for a in angles if 80 < abs(a) < 100)
        features.extend([h_lines / max(num_lines, 1), v_lines / max(num_lines, 1)])
    else:
        features.extend([0, 0, 0, 0])

    # Symmetry features
    left = gray[:, :size//2]
    right = np.flip(gray[:, size//2:], axis=1)
    min_w = min(left.shape[1], right.shape[1])
    h_sym = np.mean(np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float)))
    features.append(h_sym)

    top = gray[:size//2, :]
    bottom = np.flip(gray[size//2:, :], axis=0)
    min_h = min(top.shape[0], bottom.shape[0])
    v_sym = np.mean(np.abs(top[:min_h, :].astype(float) - bottom[:min_h, :].astype(float)))
    features.append(v_sym)

    # Noise estimation (high-frequency content)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blur.astype(float)
    features.extend([noise.mean(), noise.std(), np.abs(noise).mean()])

    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    features.append(entropy)
    features.append(hist.max())
    features.append(np.count_nonzero(hist > 0.01))

    # Gradient orientation histogram
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features.extend([mag.mean(), mag.std()])

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
            feat = extract_stat_features(str(img_path))
            features_list.append(feat)
            labels.append(label_val)

    return np.array(features_list), np.array(labels)


def train_stat_detector():
    print("Extracting statistical features...")
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
    for n_est in [100, 200, 500]:
        for max_depth in [10, 20, None]:
            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            val_pred = rf.predict(X_val)
            val_f1 = f1_score(y_val, val_pred)
            if val_f1 > best_score:
                best_score = val_f1
                best_params = {"n_estimators": n_est, "max_depth": max_depth}

    print(f"Best params: {best_params} (val F1: {best_score:.3f})")

    rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0,
    }

    print("\n--- Test Results (Statistical + RF) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Feature importance
    feature_imp = rf.feature_importances_
    top_idx = np.argsort(feature_imp)[-10:][::-1]
    print("\nTop 10 features:")
    for idx in top_idx:
        print(f"  Feature {idx}: {feature_imp[idx]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "stat_rf_results.json", "w") as f:
        json.dump({**metrics, "best_params": {k: str(v) for k, v in best_params.items()}}, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_stat_detector()
