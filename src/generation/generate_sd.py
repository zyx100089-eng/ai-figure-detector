"""
Generate fake scientific figures using Stable Diffusion (pixel-level fakes).
Runs locally — no API key or budget needed.
"""

import json
import os
import random

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "fake_pixel", "sd"
)

CHART_PROMPTS = [
    "A {chart_type} from a machine learning research paper showing {metric} of {n} different neural network architectures on {dataset}. The chart has axis labels, a legend, and uses a professional color scheme. White background.",
    "A scientific {chart_type} comparing {n} methods, with {metric} on the y-axis. Professional academic style, suitable for a top ML conference like NeurIPS or ICML. Clean axes, proper labels.",
    "A publication-quality {chart_type} from a computer science paper. Shows {metric} across {n} approaches on {dataset}. Uses matplotlib/seaborn style with error bars. White background, clean layout.",
]

CHART_TYPES = [
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


def generate_prompt() -> tuple[str, str]:
    chart_type = random.choice(CHART_TYPES)
    template = random.choice(CHART_PROMPTS)
    prompt = template.format(
        chart_type=chart_type,
        metric=random.choice(METRICS),
        n=random.randint(3, 7),
        dataset=random.choice(DATASETS),
    )
    return chart_type.replace(" ", "_"), prompt


def generate_sd_figures(
    num_figures: int = 50,
    output_dir: str | None = None,
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> list[dict]:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Stable Diffusion on {device}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    metadata = []
    print(f"Generating {num_figures} fake figures with Stable Diffusion...")

    for i in tqdm(range(num_figures), desc="SD"):
        chart_type, prompt = generate_prompt()
        figure_id = f"sd_{chart_type}_{i:04d}"
        output_path = os.path.join(output_dir, f"{figure_id}.png")

        if os.path.exists(output_path):
            continue

        try:
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]

            image.save(output_path)
            metadata.append({
                "id": figure_id,
                "chart_type": chart_type,
                "provider": "stable_diffusion",
                "model": model_id,
                "prompt": prompt,
                "path": output_path,
            })

        except Exception as e:
            print(f"  Failed {figure_id}: {e}")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata)}/{num_figures} figures")
    print(f"Saved to {output_dir}")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fake figures with Stable Diffusion")
    parser.add_argument("-n", "--num-figures", type=int, default=50)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()

    generate_sd_figures(
        num_figures=args.num_figures,
        output_dir=args.output_dir,
        model_id=args.model,
    )
