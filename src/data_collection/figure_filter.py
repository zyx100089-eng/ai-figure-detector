"""
Filter extracted figures to keep only charts/plots (bar charts, line plots,
scatter plots, heatmaps). Two-stage pipeline:
  1. Heuristic filter (fast, removes obvious junk)
  2. CLIP zero-shot classifier (accurate, keeps only real charts)
"""

import os
import shutil

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_ORIGINAL_MAX_PIXELS = Image.MAX_IMAGE_PIXELS

DEFAULT_INPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "real", "raw_extracted"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "real", "filtered"
)

CHART_LABELS = [
    "a bar chart from a scientific paper",
    "a line plot from a scientific paper",
    "a scatter plot from a scientific paper",
    "a heatmap or confusion matrix from a scientific paper",
    "a box plot from a scientific paper",
    "a histogram from a scientific paper",
]

NON_CHART_LABELS = [
    "a photograph of a real object",
    "a logo or icon",
    "an architecture diagram or flowchart",
    "a neural network diagram",
    "a segmentation mask",
    "a 3D rendering or CAD model",
    "a table of text and numbers",
    "a mathematical equation",
]


def passes_heuristic_filter(image_path: str) -> bool:
    """Fast heuristic checks to filter obvious non-charts."""
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        img = Image.open(image_path)
        w, h = img.size
    except Exception:
        return False
    finally:
        Image.MAX_IMAGE_PIXELS = _ORIGINAL_MAX_PIXELS

    if w < 250 or h < 200:
        return False

    aspect = max(w, h) / min(w, h)
    if aspect > 4.0:
        return False

    # Resize to thumbnail for fast analysis
    thumb = img.convert("RGB")
    thumb.thumbnail((256, 256))
    arr = np.array(thumb)

    # Check color diversity on the small thumbnail
    sampled = arr.reshape(-1, 3)
    unique_colors = len(np.unique(sampled, axis=0))
    if unique_colors < 30:
        return False

    # Border brightness check on thumbnail
    border = np.concatenate([
        arr[:3, :].reshape(-1, 3),
        arr[-3:, :].reshape(-1, 3),
        arr[:, :3].reshape(-1, 3),
        arr[:, -3:].reshape(-1, 3),
    ])
    if border.mean() < 100:
        return False

    return True


class CLIPChartClassifier:
    """Zero-shot chart classifier using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.all_labels = CHART_LABELS + NON_CHART_LABELS
        self.num_chart_labels = len(CHART_LABELS)

    def is_chart(self, image_path: str, threshold: float = 0.5) -> tuple[bool, float]:
        """Classify whether an image is a chart. Returns (is_chart, chart_probability)."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return False, 0.0

        inputs = self.processor(
            text=self.all_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().numpy()

        chart_prob = float(probs[:self.num_chart_labels].sum())
        return chart_prob >= threshold, chart_prob


def filter_figures(
    input_dir: str | None = None,
    output_dir: str | None = None,
    clip_threshold: float = 0.5,
    use_clip: bool = True,
) -> list[str]:
    """
    Two-stage filtering:
      1. Heuristic filter (fast pass)
      2. CLIP zero-shot classification (accurate pass)
    """
    if input_dir is None:
        input_dir = DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(image_files)} images to filter")

    # Stage 1: heuristic
    heuristic_passed = []
    for filename in tqdm(image_files, desc="Stage 1: Heuristic"):
        filepath = os.path.join(input_dir, filename)
        if passes_heuristic_filter(filepath):
            heuristic_passed.append(filename)
    print(f"  Heuristic: {len(heuristic_passed)}/{len(image_files)} passed")

    if not use_clip:
        kept = []
        for filename in heuristic_passed:
            dest = os.path.join(output_dir, filename)
            shutil.copy2(os.path.join(input_dir, filename), dest)
            kept.append(dest)
        return kept

    # Stage 2: CLIP classification
    classifier = CLIPChartClassifier()
    kept = []
    for filename in tqdm(heuristic_passed, desc="Stage 2: CLIP"):
        filepath = os.path.join(input_dir, filename)
        is_chart, prob = classifier.is_chart(filepath, threshold=clip_threshold)
        if is_chart:
            dest = os.path.join(output_dir, filename)
            shutil.copy2(filepath, dest)
            kept.append(dest)

    print(f"  CLIP: {len(kept)}/{len(heuristic_passed)} passed (threshold={clip_threshold})")
    print(f"Final: {len(kept)}/{len(image_files)} images kept")
    return kept


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter extracted figures")
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP, heuristic only")
    args = parser.parse_args()

    filter_figures(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        clip_threshold=args.threshold,
        use_clip=not args.no_clip,
    )
