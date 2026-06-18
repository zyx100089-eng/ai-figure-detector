"""
Generate fake scientific figures by asking LLMs to write matplotlib code,
then executing that code to produce the images.

Supports OpenAI (GPT-4) with easy extension to other providers.
"""

import json
import os
import time

from openai import OpenAI
from tqdm import tqdm

from src.generation.execute_generated_code import execute_code, extract_code_from_response
from src.generation.prompt_templates import generate_batch

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "fake_codegen"
)


def generate_with_gpt4(
    prompt: str,
    model: str = "gpt-4o",
    client: OpenAI | None = None,
) -> str | None:
    """Send a prompt to GPT-4 and get code back."""
    if client is None:
        client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data visualization programmer. "
                    "Write complete, self-contained Python scripts that generate "
                    "publication-quality scientific figures. Only output the code, "
                    "no explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API error: {e}")
        return None


def generate_fake_figures(
    num_figures: int = 50,
    model: str = "gpt-4o",
    provider: str = "gpt4",
    output_dir: str | None = None,
) -> list[dict]:
    """
    Generate fake scientific figures using an LLM.
    Returns list of metadata dicts for successfully generated figures.
    """
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, provider)

    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI()
    prompts = generate_batch(num_figures)
    metadata = []

    print(f"Generating {len(prompts)} fake figures with {model}...")
    for i, (chart_type, prompt) in enumerate(tqdm(prompts, desc=f"Generating ({provider})")):
        figure_id = f"{provider}_{chart_type}_{i:04d}"
        output_path = os.path.join(output_dir, f"{figure_id}.png")

        if os.path.exists(output_path):
            continue

        response = generate_with_gpt4(prompt, model=model, client=client)
        if response is None:
            continue

        code = extract_code_from_response(response)
        if code is None:
            print(f"  Could not extract code from response for {figure_id}")
            continue

        success = execute_code(code, output_path)
        if success:
            metadata.append({
                "id": figure_id,
                "chart_type": chart_type,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "code": code,
                "path": output_path,
            })

        # Rate limiting
        time.sleep(0.5)

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    success_count = len(metadata)
    print(f"\nGenerated {success_count}/{len(prompts)} figures successfully")
    print(f"Saved to {output_dir}")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fake figures with LLMs")
    parser.add_argument("-n", "--num-figures", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--provider", default="gpt4")
    parser.add_argument("-o", "--output-dir", default=None)
    args = parser.parse_args()

    generate_fake_figures(
        num_figures=args.num_figures,
        model=args.model,
        provider=args.provider,
        output_dir=args.output_dir,
    )
