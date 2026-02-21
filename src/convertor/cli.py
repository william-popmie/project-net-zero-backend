"""CLI for converting optimizer_logic results.json to Python source files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .json_to_python import write_python_files
from .report import generate_report

# Paths are relative to the project root (where this CLI is invoked from).
DEFAULT_INPUT = Path("src/optimizer_logic/output/results.json")
DEFAULT_OUTPUT = Path("src/convertor/output")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert optimizer results.json to reconstructed Python source files."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to results.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory for generated .py files (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    try:
        print(f"[*] Reading {input_path}")
        written = write_python_files(input_path, args.output)
        print(f"[OK] {len(written)} file(s) written to {args.output}")

        report_path = generate_report(input_path, args.output)
        print(f"[OK] Impact report: {report_path}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
