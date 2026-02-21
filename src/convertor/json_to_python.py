"""Convert optimizer results.json to reconstructed Python source files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def convert_json_to_python(
    json_data: dict[str, Any],
) -> dict[str, str]:
    """
    Convert optimizer results.json to Python source files.

    Args:
        json_data: Parsed results.json with a "functions" key.

    Returns:
        Dict mapping filename to Python source code string.
    """
    by_file: dict[str, list[dict]] = defaultdict(list)
    for func in json_data.get("functions", []):
        if func.get("success"):
            by_file[func["file"]].append(func)

    output_files: dict[str, str] = {}
    for source_file, funcs in by_file.items():
        filename = Path(source_file).name
        lines = [f'"""Optimized functions from {source_file}."""', ""]
        for func in funcs:
            lines.append(func["optimized_source"])
            lines.append("")
        output_files[filename] = "\n".join(lines)

    return output_files


def write_python_files(
    json_file: Path | str,
    output_dir: Path | str,
) -> dict[str, str]:
    """
    Read results.json, convert to Python source files, and write them to output_dir.

    Args:
        json_file: Path to results.json.
        output_dir: Directory to write the generated .py files.

    Returns:
        Dict mapping written file paths to their source code.
    """
    json_path = Path(json_file)
    output_path = Path(output_dir)

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    python_files = convert_json_to_python(json_data)

    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for filename, code in python_files.items():
        dest = output_path / filename
        dest.write_text(code, encoding="utf-8")
        written[str(dest)] = code
        print(f"[OK] {dest}")

    return written
