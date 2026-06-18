"""
Generate fake scientific figures using DALL-E 3 (pixel-level fakes).
These are images generated directly by the image model, not via code.
"""

import base64
import json
import os
import random
import time

import requests
from openai import OpenAI
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "fake_pixel", "dalle"
)

CHART_PROMPTS = [
    "A {chart_type} from a machine learning research paper showing {metric} of {n} different neural network architectures on {dataset}. The chart has axis labels, a legend, and uses a professional color scheme. White background.",
    "A scientific {chart_type} comparing {n} methods, with {metric} on the y-axis. Professional academic style, suitable for a top ML conference like NeurIPS or ICML. Clean axes, proper labels.",
    "A publication-quality {chart_type} from a computer science paper. Shows {metric} across {n} approaches on {dataset}. Uses matplotlib/seaborn style with error bars. White background, clean layout.",
]

CHART_TYPES_DALLE = [
    "bar chart", "grouped bar chart", "line plot", "scatter plot",
    "heatmap", "box plot", "histogram", "violin plot",
]

METRICS = [
    "accuracy (%)", "F1 score", "precision and recall", "AUC-ROC",
    "training loss over epochs", "inference latency (ms)", "BLEU score",
    "mean average precision (mAP)", "FID score", "perplexity",
]

DATASETS = [
    "CIFAR-10", "ImageNet", "COCO", "SQuAD", "GLUE benchmark",
    "WMT translation", "LibriSpeech", "Penn Treebank",
]


def generate_dalle_prompt() -> tuple[str, str]:
    """Generate a random prompt for DALL-E. Returns (chart_type, prompt)."""
    chart_type = random.choice(CHART_TYPES_DALLE)
    template = random.choice(CHART_PROMPTS)
    prompt = template.format(
        chart_type=chart_type,
        metric=random.choice(METRICS),
        n=random.randint(3, 7),
        dataset=random.choice(DATASETS),
    )
    return chart_type.replace(" ", "_"), prompt


def generate_dalle_figures(
    num_figures: int = 50,
    output_dir: str | None = None,
) -> list[dict]:
    """Generate fake figures using DALL-E 3."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    client = OpenAI()
    metadata = []

    print(f"Generating {num_figures} fake figures with DALL-E 3...")
    for i in tqdm(range(num_figures), desc="DALL-E"):
        chart_type, prompt = generate_dalle_prompt()
        figure_id = f"dalle_{chart_type}_{i:04d}"
        output_path = os.path.join(output_dir, f"{figure_id}.png")

        if os.path.exists(output_path):
            continue

        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_data = response.data[0]
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                img_bytes = base64.b64decode(image_data.b64_json)
                with open(output_path, "wb") as f:
                    f.write(img_bytes)
            elif hasattr(image_data, 'url') and image_data.url:
                img_response = requests.get(image_data.url, timeout=30)
                img_response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
            else:
                print(f"  No image data for {figure_id}")
                continue

            metadata.append({
                "id": figure_id,
                "chart_type": chart_type,
                "provider": "dalle",
                "model": "gpt-image-1",
                "prompt": prompt,
                "path": output_path,
            })

        except Exception as e:
            print(f"  Failed to generate {figure_id}: {e}")

        # Rate limiting — DALL-E 3 has strict limits
        time.sleep(1)

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata)}/{num_figures} figures successfully")
    print(f"Saved to {output_dir}")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fake figures with DALL-E")
    parser.add_argument("-n", "--num-figures", type=int, default=50)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    generate_dalle_figures(num_figures=args.num_figures, output_dir=args.output_dir)
