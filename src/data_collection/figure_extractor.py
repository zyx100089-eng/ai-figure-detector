"""
Extract figures/images from downloaded PDFs using PyMuPDF.
Filters out small images (logos, icons) and keeps only chart-sized figures.
"""

import os

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

DEFAULT_PDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "raw_pdfs"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "real", "raw_extracted"
)

MIN_WIDTH = 200
MIN_HEIGHT = 150
MAX_ASPECT_RATIO = 5.0


def extract_figures_from_pdf(pdf_path: str, output_dir: str) -> list[str]:
    """Extract images from a single PDF. Returns list of saved image paths."""
    saved = []
    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Could not open {pdf_path}: {e}")
        return saved

    img_index = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_info in images:
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if base_image is None:
                continue

            width = base_image["width"]
            height = base_image["height"]

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            aspect = max(width, height) / max(min(width, height), 1)
            if aspect > MAX_ASPECT_RATIO:
                continue

            ext = base_image["ext"]
            if ext not in ("png", "jpeg", "jpg"):
                continue

            filename = f"{paper_id}_p{page_num}_img{img_index}.png"
            filepath = os.path.join(output_dir, filename)

            image_bytes = base_image["image"]
            with open(filepath, "wb") as f:
                f.write(image_bytes)

            # Re-save as PNG for consistency
            try:
                img = Image.open(filepath).convert("RGB")
                img.save(filepath, "PNG")
                saved.append(filepath)
                img_index += 1
            except Exception:
                os.remove(filepath)

    doc.close()
    return saved


def extract_all_figures(
    pdf_dir: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Extract figures from all PDFs in a directory."""
    if pdf_dir is None:
        pdf_dir = DEFAULT_PDF_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    all_extracted = []
    for pdf_file in tqdm(pdf_files, desc="Extracting figures"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        extracted = extract_figures_from_pdf(pdf_path, output_dir)
        all_extracted.extend(extracted)

    print(f"\nExtracted {len(all_extracted)} figures to {output_dir}")
    return all_extracted


def get_extraction_stats(output_dir: str | None = None) -> dict:
    """Get basic stats about extracted figures."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if not os.path.exists(output_dir):
        return {"total": 0}

    files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    sizes = []
    for f in files:
        try:
            img = Image.open(os.path.join(output_dir, f))
            sizes.append(img.size)
        except Exception:
            continue

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    return {
        "total": len(files),
        "avg_width": sum(widths) / len(widths) if widths else 0,
        "avg_height": sum(heights) / len(heights) if heights else 0,
        "min_width": min(widths) if widths else 0,
        "max_width": max(widths) if widths else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract figures from PDFs")
    parser.add_argument(
        "-i", "--input-dir", default=DEFAULT_PDF_DIR,
        help="Directory containing PDFs",
    )
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for extracted figures",
    )
    args = parser.parse_args()

    extract_all_figures(pdf_dir=args.input_dir, output_dir=args.output_dir)

    stats = get_extraction_stats(args.output_dir)
    print(f"\nStats: {stats}")
