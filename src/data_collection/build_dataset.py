"""
Build the unified ML dataset from collected real and fake figures.
Creates metadata.csv with labels and train/val/test splits.
"""

import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "splits"


def collect_real_figures() -> list[dict]:
    filtered_dir = DATASET_DIR / "real" / "filtered"
    records = []
    for f in sorted(filtered_dir.glob("*.png")):
        records.append({
            "filename": f.name,
            "source_path": str(f),
            "label": "real",
            "fake_type": "none",
            "provider": "arxiv",
        })
    return records


def collect_fake_codegen() -> list[dict]:
    codegen_dir = DATASET_DIR / "fake_codegen" / "gpt4"
    meta_path = codegen_dir / "metadata.json"
    records = []

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for entry in meta:
            fname = Path(entry["path"]).name
            fpath = codegen_dir / fname
            if fpath.exists():
                records.append({
                    "filename": fname,
                    "source_path": str(fpath),
                    "label": "fake",
                    "fake_type": "codegen",
                    "provider": entry.get("provider", "gpt4"),
                    "chart_type": entry.get("chart_type", "unknown"),
                })
    else:
        for f in sorted(codegen_dir.glob("*.png")):
            records.append({
                "filename": f.name,
                "source_path": str(f),
                "label": "fake",
                "fake_type": "codegen",
                "provider": "gpt4",
            })
    return records


def collect_fake_pixel() -> list[dict]:
    records = []
    for provider_dir in (DATASET_DIR / "fake_pixel").iterdir():
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        meta_path = provider_dir / "metadata.json"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for entry in meta:
                fname = Path(entry["path"]).name
                fpath = provider_dir / fname
                if fpath.exists():
                    records.append({
                        "filename": fname,
                        "source_path": str(fpath),
                        "label": "fake",
                        "fake_type": "pixel",
                        "provider": provider,
                        "chart_type": entry.get("chart_type", "unknown"),
                    })
        else:
            for f in sorted(provider_dir.glob("*.png")):
                records.append({
                    "filename": f.name,
                    "source_path": str(f),
                    "label": "fake",
                    "fake_type": "pixel",
                    "provider": provider,
                })
    return records


def balance_dataset(df: pd.DataFrame, max_ratio: float = 2.0) -> pd.DataFrame:
    """Downsample the majority class so it's at most max_ratio × minority."""
    counts = df["label"].value_counts()
    minority_count = counts.min()
    max_count = int(minority_count * max_ratio)

    balanced = []
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        if len(subset) > max_count:
            subset = subset.sample(n=max_count, random_state=42)
        balanced.append(subset)
    return pd.concat(balanced, ignore_index=True)


def build_dataset(
    balance: bool = True,
    max_ratio: float = 2.0,
    test_size: float = 0.15,
    val_size: float = 0.15,
    copy_files: bool = True,
):
    print("Collecting figures...")
    real = collect_real_figures()
    fake_code = collect_fake_codegen()
    fake_pixel = collect_fake_pixel()

    print(f"  Real figures: {len(real)}")
    print(f"  Code-level fakes: {len(fake_code)}")
    print(f"  Pixel-level fakes: {len(fake_pixel)}")

    all_records = real + fake_code + fake_pixel
    df = pd.DataFrame(all_records)

    if balance and len(df[df["label"] == "fake"]) > 0:
        print(f"\nBalancing dataset (max ratio {max_ratio}:1)...")
        df = balance_dataset(df, max_ratio=max_ratio)
        print(f"  After balancing: {df['label'].value_counts().to_dict()}")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_val["label"],
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    df_final = pd.concat([train, val, test], ignore_index=True)

    print(f"\nSplit sizes:")
    for split in ["train", "val", "test"]:
        split_df = df_final[df_final["split"] == split]
        print(f"  {split}: {len(split_df)} "
              f"(real={len(split_df[split_df['label']=='real'])}, "
              f"fake={len(split_df[split_df['label']=='fake'])})")

    if copy_files:
        print("\nCopying files to split directories...")
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                split_dir = OUTPUT_DIR / split / label
                split_dir.mkdir(parents=True, exist_ok=True)

            split_df = df_final[df_final["split"] == split]
            for _, row in split_df.iterrows():
                src = Path(row["source_path"])
                dst = OUTPUT_DIR / split / row["label"] / row["filename"]
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

    csv_path = DATASET_DIR / "metadata.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"\nMetadata saved to {csv_path}")
    print(f"Total dataset size: {len(df_final)}")
    return df_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build unified ML dataset")
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    build_dataset(
        balance=not args.no_balance,
        max_ratio=args.max_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        copy_files=not args.no_copy,
    )
