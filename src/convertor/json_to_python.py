"""Convert results.json (function-test pairs) to runnable Python test files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_json_to_python(
    json_data: dict[str, Any],
    format_style: str = "function",
) -> dict[str, str]:
    """
    Convert function-test pairs JSON to Python code strings.

    Args:
        json_data: Parsed results.json with a "pairs" key.
        format_style: "function" (standalone functions) or "class" (grouped in a class).

    Returns:
        Dict mapping filename to Python source code string.
    """
    output_files: dict[str, str] = {}

    for pair in json_data.get("pairs", []):
        function_name = pair["function_name"]
        tests = pair["tests"]

        if format_style == "function":
            code = _generate_standalone_tests(function_name, tests)
            filename = f"test_{function_name}.py"
        elif format_style == "class":
            code = _generate_class_tests(function_name, tests)
            filename = f"test_{function_name}_class.py"
        else:
            raise ValueError(f"Unknown format_style: {format_style!r}. Choose 'function' or 'class'.")

        output_files[filename] = code

    return output_files


def write_python_files(
    json_file: Path | str,
    output_dir: Path | str,
    format_style: str = "function",
) -> dict[str, str]:
    """
    Read results.json, convert to Python test files, and write them to output_dir.

    Args:
        json_file: Path to results.json.
        output_dir: Directory to write the generated .py files.
        format_style: "function" or "class".

    Returns:
        Dict mapping written file paths to their source code.
    """
    json_path = Path(json_file)
    output_path = Path(output_dir)

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    python_files = convert_json_to_python(json_data, format_style)

    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for filename, code in python_files.items():
        dest = output_path / filename
        dest.write_text(code, encoding="utf-8")
        written[str(dest)] = code
        print(f"[OK] {dest}")

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_standalone_tests(function_name: str, tests: list[dict[str, str]]) -> str:
    lines = [
        '"""Auto-generated test module."""',
        "",
        "# TODO: update import to match your project structure",
        f"# from src.app.my_module import {function_name}",
        "",
    ]
    for test in tests:
        lines.append(_method_to_function(test["test_code"]))
        lines.append("")
    return "\n".join(lines)


def _generate_class_tests(function_name: str, tests: list[dict[str, str]]) -> str:
    class_name = f"Test{function_name.capitalize()}"
    lines = [
        '"""Auto-generated test module."""',
        "",
        "# TODO: update import to match your project structure",
        f"# from src.app.my_module import {function_name}",
        "",
        f"class {class_name}:",
        f'    """Test suite for {function_name}."""',
        "",
    ]
    if not tests:
        lines.append("    pass")
    else:
        for test in tests:
            for line in test["test_code"].split("\n"):
                lines.append("    " + line if line else "")
            lines.append("")
    return "\n".join(lines)


def _method_to_function(method_code: str) -> str:
    """Convert a test method (with self) to a standalone test function."""
    lines = method_code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i == 0 and "def " in line and "(self)" in line:
            result.append(line.replace("(self)", "()").replace("(self, ", "("))
        elif i > 0 and line.startswith("    "):
            result.append(line[4:])
        else:
            result.append(line)
    return "\n".join(result)
