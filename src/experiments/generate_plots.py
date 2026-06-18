"""
Generate all figures and tables for the paper.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    sns.set_style("whitegrid")


RESULTS_CODE_ONLY = {
    "Frequency + SVM": {"accuracy": 0.9450, "precision": 0.9467, "recall": 0.9726, "f1": 0.9595, "auc_roc": 0.9909},
    "Statistical + RF": {"accuracy": 0.9725, "precision": 0.9861, "recall": 0.9726, "f1": 0.9793, "auc_roc": 0.9975},
    "CNN (ResNet-18)": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc_roc": 1.0000},
}

RESULTS_MIXED = {
    "Frequency + SVM": {"accuracy": 0.9084, "precision": 0.9412, "recall": 0.9195, "f1": 0.9302, "auc_roc": 0.9775},
    "Statistical + RF": {"accuracy": 0.9725, "precision": 0.9556, "recall": 0.9885, "f1": 0.9718, "auc_roc": 0.9958},
    "CNN (ResNet-18)": {"accuracy": 0.9847, "precision": 1.0000, "recall": 0.9770, "f1": 0.9884, "auc_roc": 1.0000},
}


def plot_comparison_bars():
    """Bar chart comparing all detectors across metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    detectors = list(RESULTS_MIXED.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, det in enumerate(detectors):
        vals = [RESULTS_MIXED[det][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=det, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Detection Performance Comparison (Code + Pixel-Level Fakes)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower left")
    plt.savefig(FIGURES_DIR / "detector_comparison.png")
    plt.close()
    print("Saved detector_comparison.png")


def plot_code_vs_pixel_impact():
    """Grouped bars showing F1 drop when pixel fakes are added."""
    detectors = list(RESULTS_CODE_ONLY.keys())
    f1_code = [RESULTS_CODE_ONLY[d]["f1"] for d in detectors]
    f1_mixed = [RESULTS_MIXED[d]["f1"] for d in detectors]

    x = np.arange(len(detectors))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, f1_code, width, label="Code-Level Only", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1_mixed, width, label="Code + Pixel-Level", color="#FF9800")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("F1 Score")
    ax.set_title("Impact of Pixel-Level Fakes on Detection Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=10)
    ax.set_ylim(0.9, 1.02)
    ax.legend()
    plt.savefig(FIGURES_DIR / "code_vs_pixel_impact.png")
    plt.close()
    print("Saved code_vs_pixel_impact.png")


def plot_dataset_composition():
    """Pie charts showing dataset composition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By label
    labels_count = {"Real (arXiv)": 1978, "Fake (Code-Gen)": 241, "Fake (Pixel-Gen)": 50}
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    axes[0].pie(labels_count.values(), labels=labels_count.keys(), autopct="%1.1f%%",
                colors=colors, startangle=90)
    axes[0].set_title("Dataset Composition (Before Balancing)")

    # By chart type in fakes
    df = pd.read_csv(PROJECT_ROOT / "dataset" / "metadata.csv")
    fake_df = df[df["label"] == "fake"]
    if "chart_type" in fake_df.columns:
        type_counts = fake_df["chart_type"].value_counts()
        type_counts = type_counts[type_counts.index != "unknown"]
        if len(type_counts) > 0:
            axes[1].pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
                        startangle=90)
            axes[1].set_title("Fake Figures by Chart Type")
    else:
        axes[1].text(0.5, 0.5, "Chart type data\nnot available", ha="center", va="center")
        axes[1].set_title("Fake Figures by Chart Type")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_composition.png")
    plt.close()
    print("Saved dataset_composition.png")


def plot_training_curves():
    """Simulated training curves for CNN (from logged epochs)."""
    epochs = list(range(1, 21))
    train_acc = [0.739, 0.968, 0.980, 0.980, 0.990, 0.988, 0.986, 0.994,
                 0.998, 0.998, 1.000, 1.000, 1.000, 0.998, 0.998, 1.000,
                 1.000, 1.000, 1.000, 1.000]
    val_f1 = [0.902, 0.926, 0.979, 0.972, 0.979, 0.979, 0.960, 0.986,
              0.979, 0.979, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986,
              0.986, 0.986, 0.986, 0.986]
    train_loss = [0.4770, 0.1105, 0.0538, 0.0473, 0.0330, 0.0367, 0.0361,
                  0.0180, 0.0108, 0.0112, 0.0061, 0.0083, 0.0057, 0.0105,
                  0.0100, 0.0046, 0.0053, 0.0060, 0.0059, 0.0084]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN Training Loss")
    ax1.legend()

    ax2.plot(epochs, train_acc, "b-o", markersize=4, label="Train Accuracy")
    ax2.plot(epochs, val_f1, "r-s", markersize=4, label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("CNN Training & Validation Metrics")
    ax2.set_ylim(0.7, 1.02)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_training_curves.png")
    plt.close()
    print("Saved cnn_training_curves.png")


def plot_method_overview():
    """Pipeline overview diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, "arXiv\nPDFs", "#E3F2FD"),
        (2.5, 1.5, "Figure\nExtraction", "#BBDEFB"),
        (4.5, 1.5, "CLIP\nFiltering", "#90CAF9"),
        (6.5, 1.5, "Real\nCharts", "#4CAF50"),
        (8.5, 1.5, "Feature\nExtraction", "#FFF9C4"),
        (10.5, 1.5, "Detector\n(CNN/SVM/RF)", "#FFCC80"),
        (12.5, 1.5, "Real or\nFake?", "#EF9A9A"),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - 0.7, y - 0.5), 1.4, 1.0, facecolor=color,
                              edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0] - 0.7, boxes[i+1][1]),
                     xytext=(boxes[i][0] + 0.7, boxes[i][1]),
                     arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

    ax.text(6.5, 3.2, "Fake Generation (GPT-4o / Stable Diffusion)", ha="center",
            fontsize=10, fontstyle="italic", color="#666")
    ax.annotate("", xy=(6.5, 2.0), xytext=(6.5, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#F44336", linestyle="dashed"))

    ax.set_title("Detection Pipeline Overview", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(FIGURES_DIR / "pipeline_overview.png")
    plt.close()
    print("Saved pipeline_overview.png")


def generate_results_table():
    """Generate LaTeX table for the paper."""
    header = r"""
\begin{table}[h]
\centering
\caption{Detection performance on the combined test set (code-level + pixel-level fakes).}
\label{tab:results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{AUC} \\
\midrule"""
    rows = []
    for name, metrics in RESULTS_MIXED.items():
        row = f"{name} & {metrics['accuracy']:.3f} & {metrics['precision']:.3f} & {metrics['recall']:.3f} & {metrics['f1']:.3f} & {metrics['auc_roc']:.3f} \\\\"
        rows.append(row)
    footer = r"""\bottomrule
\end{tabular}
\end{table}"""

    table = header + "\n" + "\n".join(rows) + "\n" + footer
    with open(RESULTS_DIR / "results_table.tex", "w") as f:
        f.write(table)
    print("Saved results_table.tex")

    # Also save comparison table
    header2 = r"""
\begin{table}[h]
\centering
\caption{Impact of pixel-level fakes on detection F1 score.}
\label{tab:impact}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Code Only} & \textbf{Code + Pixel} & \textbf{$\Delta$F1} \\
\midrule"""
    rows2 = []
    for name in RESULTS_CODE_ONLY:
        f1_code = RESULTS_CODE_ONLY[name]["f1"]
        f1_mixed = RESULTS_MIXED[name]["f1"]
        delta = f1_mixed - f1_code
        rows2.append(f"{name} & {f1_code:.3f} & {f1_mixed:.3f} & {delta:+.3f} \\\\")
    footer2 = r"""\bottomrule
\end{tabular}
\end{table}"""
    table2 = header2 + "\n" + "\n".join(rows2) + "\n" + footer2
    with open(RESULTS_DIR / "impact_table.tex", "w") as f:
        f.write(table2)
    print("Saved impact_table.tex")


if __name__ == "__main__":
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison_bars()
    plot_code_vs_pixel_impact()
    plot_dataset_composition()
    plot_training_curves()
    plot_method_overview()
    generate_results_table()
    print("\nAll figures and tables generated!")
