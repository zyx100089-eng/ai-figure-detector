"""
Demo: detect whether a scientific figure is real or AI-generated.

Usage:
    python demo.py path/to/figure.png
    python demo.py path/to/figure.png --method all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.detection.cnn_detector import build_model
from src.detection.frequency_detector import extract_frequency_features
from src.detection.stat_detector import extract_stat_features

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

    # ImageFolder sorts: fake=0, real=1
    real_prob = probs[1].item()
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    confidence = real_prob if label == "REAL" else 1 - real_prob
    return label, confidence


def predict_frequency(image_path: str) -> tuple[str, float]:
    features = extract_frequency_features(image_path)
    energy_ratio = features[12:16].mean() / (features[0:4].mean() + 1e-10)
    fake_score = min(1.0, max(0.0, energy_ratio * 2))
    label = "FAKE" if fake_score > 0.5 else "REAL"
    return label, max(fake_score, 1 - fake_score)


def predict_stat(image_path: str) -> tuple[str, float]:
    features = extract_stat_features(image_path)
    color_diversity = features[15] if len(features) > 15 else 0.5
    edge_density = features[16] if len(features) > 16 else 0.5
    heuristic_score = (color_diversity + edge_density) / 2
    label = "REAL" if heuristic_score > 0.3 else "FAKE"
    return label, 0.7


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
        "frequency": ("Frequency Analysis", lambda p: predict_frequency(p)),
        "stat": ("Statistical Analysis", lambda p: predict_stat(p)),
    }

    if args.method == "all":
        to_run = methods
    else:
        to_run = {args.method: methods[args.method]}

    results = {}
    for key, (name, func) in to_run.items():
        if key == "cnn":
            label, conf = func(args.image)
        else:
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
