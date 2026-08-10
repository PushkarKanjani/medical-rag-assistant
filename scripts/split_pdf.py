"""
split_pdf.py
────────────
Splits a large PDF in the data folder into 4 separate smaller PDF files
(each target size < 20 MB) and saves them in the data/ directory.

Usage:
    .venv\\Scripts\\python.exe scripts/split_pdf.py
    or
    python scripts/split_pdf.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path


def get_pdf_reader_writer():
    """Import and return PdfReader and PdfWriter from pypdf or PyPDF2."""
    try:
        from pypdf import PdfReader, PdfWriter
        return PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
            return PdfReader, PdfWriter
        except ImportError:
            print("ERROR: Neither 'pypdf' nor 'PyPDF2' is installed.")
            print("Please run: pip install pypdf")
            sys.exit(1)


def split_pdf(
    input_pdf_path: Path | str,
    output_dir: Path | str | None = None,
    num_parts: int = 4,
    max_size_mb: float = 20.0,
) -> list[Path]:
    """
    Splits the given PDF into `num_parts` separate files and saves them to `output_dir`.

    Args:
        input_pdf_path: Path to the source PDF file.
        output_dir: Destination folder for split files (defaults to same folder as input).
        num_parts: Number of parts to divide the PDF into (default: 4).
        max_size_mb: Target maximum size per split file in MB (for validation report).

    Returns:
        List of Path objects for the created PDF files.
    """
    input_path = Path(input_pdf_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Source PDF file not found at: {input_path}")

    out_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    PdfReader, PdfWriter = get_pdf_reader_writer()

    total_size_mb = input_path.stat().st_size / (1024 * 1024)
    print("=" * 65)
    print(f"Source PDF   : {input_path.name}")
    print(f"Source Path  : {input_path}")
    print(f"Total Size   : {total_size_mb:.2f} MB")
    print(f"Target Parts : {num_parts} (target < {max_size_mb:.1f} MB each)")
    print("=" * 65)

    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    print(f"Total Pages  : {total_pages}\n")

    pages_per_part = math.ceil(total_pages / num_parts)
    created_files: list[Path] = []
    base_name = input_path.stem

    for part_idx in range(num_parts):
        start_page = part_idx * pages_per_part
        end_page = min((part_idx + 1) * pages_per_part, total_pages)

        if start_page >= total_pages:
            break

        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        output_filename = f"{base_name}_part{part_idx + 1}.pdf"
        output_filepath = out_dir / output_filename

        print(f"Writing Part {part_idx + 1}/{num_parts}: pages {start_page + 1} to {end_page} ({end_page - start_page} pages)...")
        with open(output_filepath, "wb") as f_out:
            writer.write(f_out)

        part_size_mb = output_filepath.stat().st_size / (1024 * 1024)
        status = "OK" if part_size_mb <= max_size_mb else "WARNING (> 20 MB)"
        print(f"  -> Saved: {output_filepath.name} ({part_size_mb:.2f} MB) [{status}]")
        created_files.append(output_filepath)

    print("\n" + "=" * 65)
    print(f"Successfully generated {len(created_files)} PDF parts in: {out_dir}")
    print("=" * 65)
    for i, file_path in enumerate(created_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Part {i}: {file_path.name} - {size_mb:.2f} MB")
    print("=" * 65)

    return created_files


def find_default_pdf(data_dir: Path) -> Path:
    """Find the target PDF in data directory."""
    preferred_name = "The_Gale_Encyclopedia_of_Medicine_3rd_Edition.pdf"
    preferred_path = data_dir / preferred_name
    if preferred_path.exists():
        return preferred_path

    # Fallback to any .pdf in data_dir
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    # Exclude already split parts if running again
    unsplit = [p for p in pdf_files if not p.stem.endswith(("_part1", "_part2", "_part3", "_part4"))]
    if unsplit:
        return max(unsplit, key=lambda p: p.stat().st_size)
    return max(pdf_files, key=lambda p: p.stat().st_size)


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir = workspace_root / "data"

    if len(sys.argv) > 1:
        input_pdf = Path(sys.argv[1])
    else:
        try:
            input_pdf = find_default_pdf(data_dir)
        except Exception as e:
            print(f"Error locating input PDF: {e}")
            sys.exit(1)

    split_pdf(
        input_pdf_path=input_pdf,
        output_dir=data_dir,
        num_parts=4,
        max_size_mb=20.0,
    )


if __name__ == "__main__":
    main()
