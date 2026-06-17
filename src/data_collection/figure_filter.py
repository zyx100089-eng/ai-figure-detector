"""
Filter extracted figures to keep only charts/plots (bar charts, line plots,
scatter plots, heatmaps). Uses heuristics based on image properties, with an
optional CLIP-based classifier for higher accuracy.

For now: heuristic filtering + manual review.
Later: add CLIP zero-shot classification.
"""

import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

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
]

NON_CHART_LABELS = [
    "a photograph",
    "a logo or icon",
    "an architecture diagram or flowchart",
    "a neural network diagram",
    "a segmentation mask",
    "a natural image",
    "a 3D rendering",
]


def passes_heuristic_filter(image_path: str) -> bool:
    """Basic heuristic checks to filter obvious non-charts."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    w, h = img.size

    # Too small — likely an icon or logo
    if w < 250 or h < 200:
        return False

    # Extreme aspect ratio — likely a banner or separator
    aspect = max(w, h) / min(w, h)
    if aspect > 4.0:
        return False

    arr = np.array(img)

    # Mostly single color — likely a mask or blank
    unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
    if unique_colors < 50:
        return False

    # Charts tend to have a white/light background
    # Check if the border region is mostly light
    border = np.concatenate([
        arr[:10, :].reshape(-1, 3),     # top
        arr[-10:, :].reshape(-1, 3),    # bottom
        arr[:, :10].reshape(-1, 3),     # left
        arr[:, -10:].reshape(-1, 3),    # right
    ])
    mean_border_brightness = border.mean()
    if mean_border_brightness < 100:
        return False

    return True


def filter_figures_heuristic(
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Apply heuristic filters and copy passing images to output dir."""
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

    kept = []
    for filename in tqdm(image_files, desc="Filtering"):
        filepath = os.path.join(input_dir, filename)
        if passes_heuristic_filter(filepath):
            dest = os.path.join(output_dir, filename)
            shutil.copy2(filepath, dest)
            kept.append(dest)

    print(f"Kept {len(kept)}/{len(image_files)} images after heuristic filtering")
    return kept


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter extracted figures")
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    filter_figures_heuristic(input_dir=args.input_dir, output_dir=args.output_dir)
