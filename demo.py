"""
Demo: detect whether a scientific figure is real or AI-generated.

Usage:
    python demo.py path/to/figure.png
    python demo.py path/to/figure.png --method all
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torchvision import transforms

from src.detection.cnn_detector import build_model
from src.detection.frequency_detector import extract_frequency_features, load_split as load_freq_split
from src.detection.stat_detector import extract_stat_features, load_split as load_stat_split

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def predict_cnn(image_path: str, device: str = "cpu") -> tuple[str, float]:
    model_path = RESULTS_DIR / "cnn_resnet18.pth"
    if not model_path.exists():
        return "unknown", 0.0

    model = build_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    real_prob = probs[1].item()
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    confidence = real_prob if label == "REAL" else 1 - real_prob
    return label, confidence


def predict_frequency(image_path: str) -> tuple[str, float]:
    features = extract_frequency_features(image_path).reshape(1, -1)

    X_train, y_train = load_freq_split("train")
    X_val, y_val = load_freq_split("val")
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_val]))
    features = scaler.transform(features)

    svm = SVC(C=10.0, gamma="scale", kernel="rbf", probability=True)
    svm.fit(scaler.transform(np.vstack([X_train, X_val])), np.concatenate([y_train, y_val]))

    prob = svm.predict_proba(features)[0, 1]
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if label == "REAL" else 1 - prob
    return label, confidence


def predict_stat(image_path: str) -> tuple[str, float]:
    features = extract_stat_features(image_path).reshape(1, -1)

    X_train, y_train = load_stat_split("train")
    X_val, y_val = load_stat_split("val")
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_val]))
    features = scaler.transform(features)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(scaler.transform(np.vstack([X_train, X_val])), np.concatenate([y_train, y_val]))

    prob = rf.predict_proba(features)[0, 1]
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if label == "REAL" else 1 - prob
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated scientific figures")
    parser.add_argument("image", help="Path to the figure image")
    parser.add_argument("--method", choices=["cnn", "frequency", "stat", "all"], default="cnn")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: {args.image} not found")
        sys.exit(1)

    print(f"\nAnalyzing: {args.image}\n")

    methods = {
        "cnn": ("CNN (ResNet-18)", predict_cnn),
        "frequency": ("Frequency Analysis", predict_frequency),
        "stat": ("Statistical Analysis", predict_stat),
    }

    if args.method == "all":
        to_run = methods
    else:
        to_run = {args.method: methods[args.method]}

    results = {}
    for key, (name, func) in to_run.items():
        label, conf = func(args.image)
        results[key] = (label, conf)
        print(f"  {name:25s}  {label:4s}  (confidence: {conf:.1%})")

    if len(results) > 1:
        fake_votes = sum(1 for l, _ in results.values() if l == "FAKE")
        verdict = "FAKE" if fake_votes > len(results) / 2 else "REAL"
        print(f"\n  {'Ensemble Verdict':25s}  {verdict}")

    print()


if __name__ == "__main__":
    main()
