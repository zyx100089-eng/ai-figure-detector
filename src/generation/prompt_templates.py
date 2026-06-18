"""
Prompt templates for generating fake scientific figures via LLMs.
Each template asks the LLM to produce matplotlib/seaborn code that creates
a realistic-looking scientific chart with plausible (but fabricated) data.
"""

import random

CHART_TYPES = ["bar_chart", "line_plot", "scatter_plot", "heatmap", "box_plot", "histogram"]

METRICS = [
    "accuracy", "F1 score", "precision", "recall", "AUC-ROC",
    "BLEU score", "perplexity", "loss", "RMSE", "MAE",
    "inference time (ms)", "throughput (samples/s)", "memory usage (GB)",
    "FID score", "PSNR", "SSIM", "mAP", "IoU", "top-1 accuracy", "top-5 accuracy",
]

DATASETS = [
    "CIFAR-10", "CIFAR-100", "ImageNet", "COCO", "MNIST",
    "SST-2", "SQuAD", "GLUE", "SuperGLUE", "WMT14",
    "LibriSpeech", "Common Voice", "Penn Treebank", "WikiText-103",
    "NYU Depth V2", "KITTI", "ModelNet40", "ShapeNet",
]

METHODS = [
    "ResNet-50", "ViT-B/16", "BERT-base", "GPT-2", "RoBERTa",
    "EfficientNet-B0", "DenseNet-121", "MobileNetV3", "Swin-T", "ConvNeXt-T",
    "XGBoost", "Random Forest", "LightGBM", "SVM", "KNN",
    "LSTM", "GRU", "Transformer", "U-Net", "YOLO v8",
]

STYLE_INSTRUCTIONS = [
    "Use a clean, professional style suitable for a top-tier ML conference paper.",
    "Use seaborn's 'whitegrid' style with a muted color palette.",
    "Use matplotlib's default style with a color scheme from a Nature paper.",
    "Use a minimal style with thin lines and no grid, similar to ICML papers.",
    "Use a professional style with gridlines and high contrast colors.",
]


def _pick(lst: list, n: int) -> list:
    return random.sample(lst, min(n, len(lst)))


def bar_chart_prompt() -> str:
    methods = _pick(METHODS, random.randint(4, 7))
    metric = random.choice(METRICS)
    dataset = random.choice(DATASETS)
    style = random.choice(STYLE_INSTRUCTIONS)
    return f"""Write a complete Python script using matplotlib that generates a bar chart comparing the following methods on {dataset}:
Methods: {', '.join(methods)}
Metric: {metric}

Requirements:
- Generate realistic but fictional numerical values for each method
- Include error bars
- Label axes clearly with appropriate font sizes
- {style}
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


def line_plot_prompt() -> str:
    methods = _pick(METHODS, random.randint(3, 5))
    metric = random.choice(METRICS)
    style = random.choice(STYLE_INSTRUCTIONS)
    x_axis = random.choice(["training epochs", "number of parameters (M)", "dataset size (%)", "learning rate", "batch size"])
    return f"""Write a complete Python script using matplotlib that generates a line plot showing {metric} vs {x_axis} for:
Methods: {', '.join(methods)}

Requirements:
- Generate realistic but fictional data with 8-15 data points per method
- Include confidence intervals or shaded error regions
- Use distinct line styles (solid, dashed, dotted) and markers
- {style}
- Add a legend in the best location
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


def scatter_plot_prompt() -> str:
    metric_x = random.choice(METRICS[:10])
    metric_y = random.choice(METRICS[10:])
    methods = _pick(METHODS, random.randint(4, 8))
    style = random.choice(STYLE_INSTRUCTIONS)
    return f"""Write a complete Python script using matplotlib that generates a scatter plot comparing {metric_x} (x-axis) vs {metric_y} (y-axis) for different models:
Models: {', '.join(methods)}

Requirements:
- Generate realistic but fictional (x, y) values for each model
- Use different colors and markers for each model
- Annotate each point with the model name
- {style}
- Optionally add a Pareto frontier line
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


def heatmap_prompt() -> str:
    n = random.randint(5, 10)
    methods = _pick(METHODS, n)
    datasets = _pick(DATASETS, random.randint(4, 6))
    metric = random.choice(METRICS[:5])
    style = random.choice(STYLE_INSTRUCTIONS)
    return f"""Write a complete Python script using matplotlib and seaborn that generates a heatmap (confusion matrix style) showing {metric} across methods and datasets:
Methods (rows): {', '.join(methods)}
Datasets (columns): {', '.join(datasets)}

Requirements:
- Generate realistic but fictional values between 0 and 1
- Annotate each cell with the value
- Use an appropriate colormap (e.g., 'YlOrRd', 'viridis', or 'Blues')
- {style}
- Bold the best result in each column
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


def box_plot_prompt() -> str:
    methods = _pick(METHODS, random.randint(4, 6))
    metric = random.choice(METRICS)
    dataset = random.choice(DATASETS)
    style = random.choice(STYLE_INSTRUCTIONS)
    return f"""Write a complete Python script using matplotlib that generates a box plot comparing the distribution of {metric} across different methods on {dataset}:
Methods: {', '.join(methods)}

Requirements:
- Generate realistic but fictional distributions (20-50 samples per method)
- Show individual data points as scatter overlay
- Use different colors for each box
- {style}
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


def histogram_prompt() -> str:
    metric = random.choice(METRICS)
    style = random.choice(STYLE_INSTRUCTIONS)
    comparison = random.choice(["before and after fine-tuning", "with and without data augmentation", "on clean vs noisy data"])
    return f"""Write a complete Python script using matplotlib that generates a histogram comparing the distribution of {metric} {comparison}:

Requirements:
- Generate realistic but fictional data (200-500 samples per distribution)
- Use semi-transparent overlapping histograms
- Add vertical lines for means with annotations
- Include a legend
- {style}
- Save the figure to 'output.png' with dpi=150 and tight bounding box
- Do NOT use plt.show()
- The chart should look like it belongs in a real ML research paper"""


PROMPT_GENERATORS = {
    "bar_chart": bar_chart_prompt,
    "line_plot": line_plot_prompt,
    "scatter_plot": scatter_plot_prompt,
    "heatmap": heatmap_prompt,
    "box_plot": box_plot_prompt,
    "histogram": histogram_prompt,
}


def generate_prompt(chart_type: str | None = None) -> tuple[str, str]:
    """Generate a random prompt. Returns (chart_type, prompt_text)."""
    if chart_type is None:
        chart_type = random.choice(CHART_TYPES)
    prompt = PROMPT_GENERATORS[chart_type]()
    return chart_type, prompt


def generate_batch(n: int = 50) -> list[tuple[str, str]]:
    """Generate n prompts with balanced chart types."""
    per_type = max(1, n // len(CHART_TYPES))
    prompts = []
    for chart_type in CHART_TYPES:
        for _ in range(per_type):
            prompts.append(generate_prompt(chart_type))
    random.shuffle(prompts)
    return prompts[:n]
