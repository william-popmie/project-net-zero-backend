"""CLI for rewriting optimized functions back into a project directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inplace_rewriter import rewrite_functions_inplace

DEFAULT_INPUT = Path("src/optimizer_logic/output/results.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite optimized functions in-place in the project directory."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to optimizer results.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-p",
        "--project-dir",
        required=True,
        help="Project directory whose source files will be rewritten.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        print(f"[ERROR] Project directory not found: {project_dir}")
        sys.exit(1)

    try:
        print(f"[*] Reading {input_path}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        modified = rewrite_functions_inplace(project_dir, data.get("functions", []))
        print(f"[OK] {len(modified)} file(s) updated in {project_dir}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
