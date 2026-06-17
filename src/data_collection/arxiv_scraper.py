"""
Download PDFs from arXiv in specific categories (cs.LG, cs.CV, stat.ML).
Uses the arXiv API to search for recent papers, then downloads PDFs.
"""

import os
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

ARXIV_API_URL = "http://export.arxiv.org/api/query"
PDF_BASE_URL = "https://arxiv.org/pdf/"

CATEGORIES = ["cs.LG", "cs.CV", "stat.ML"]

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "raw_pdfs"
)


def search_arxiv(category: str, max_results: int = 100, start: int = 0) -> list[dict]:
    """Query the arXiv API for papers in a category. Returns list of paper metadata."""
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")

        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"]
                break

        if pdf_link is None:
            pdf_link = f"{PDF_BASE_URL}{paper_id}"

        papers.append({
            "id": paper_id,
            "title": title,
            "pdf_url": pdf_link,
            "category": category,
        })

    return papers


def download_pdf(paper: dict, output_dir: str) -> str | None:
    """Download a single PDF. Returns the filepath if successful."""
    safe_id = paper["id"].replace("/", "_")
    filepath = os.path.join(output_dir, f"{safe_id}.pdf")

    if os.path.exists(filepath):
        return filepath

    pdf_url = paper["pdf_url"]
    if not pdf_url.endswith(".pdf"):
        pdf_url += ".pdf"

    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("content-type", ""):
            print(f"  Skipping {paper['id']}: not a PDF response")
            return None

        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath

    except requests.RequestException as e:
        print(f"  Failed to download {paper['id']}: {e}")
        return None


def scrape_arxiv(
    categories: list[str] | None = None,
    papers_per_category: int = 100,
    output_dir: str | None = None,
) -> list[str]:
    """
    Main function: search arXiv for papers and download their PDFs.
    Returns list of downloaded file paths.
    """
    if categories is None:
        categories = CATEGORIES
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    all_downloaded = []

    for category in categories:
        print(f"\nSearching arXiv for {category} papers...")
        papers = search_arxiv(category, max_results=papers_per_category)
        print(f"  Found {len(papers)} papers")

        # arXiv API rate limit: 1 request per 3 seconds
        time.sleep(3)

        print(f"  Downloading PDFs...")
        for paper in tqdm(papers, desc=f"  {category}"):
            filepath = download_pdf(paper, output_dir)
            if filepath:
                all_downloaded.append(filepath)
            time.sleep(1)  # be polite to arXiv servers

    print(f"\nDone. Downloaded {len(all_downloaded)} PDFs to {output_dir}")
    return all_downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download papers from arXiv")
    parser.add_argument(
        "-n", "--num-papers", type=int, default=50,
        help="Number of papers per category (default: 50)",
    )
    parser.add_argument(
        "-c", "--categories", nargs="+", default=CATEGORIES,
        help=f"arXiv categories (default: {CATEGORIES})",
    )
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for PDFs",
    )
    args = parser.parse_args()

    scrape_arxiv(
        categories=args.categories,
        papers_per_category=args.num_papers,
        output_dir=args.output_dir,
    )
