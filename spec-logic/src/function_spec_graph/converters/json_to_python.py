"""Convert function-test pair JSON to runnable Python test files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_function_code(test_code: str) -> str:
    """Extract just the function body from test code (remove 'def ...' line if it's a method)."""
    lines = test_code.split("\n")
    
    # If it's a method (has 'self' parameter), extract the implementation
    if lines and "def " in lines[0] and "(self)" in lines[0]:
        # Find the first indented line after def
        impl_lines = []
        for i, line in enumerate(lines[1:], 1):
            if line.strip():  # Skip empty lines at start
                impl_lines.append(line)
        return "\n".join(impl_lines)
    
    return test_code


def convert_json_to_python(
    json_data: dict[str, Any],
    format_style: str = "function",
) -> dict[str, str]:
    """
    Convert function-test pairs JSON to Python code strings.
    
    Args:
        json_data: The parsed function_test_pairs.json
        format_style: "function" (standalone), "class" (grouped), "both" (combined)
    
    Returns:
        Dict mapping file paths to Python code strings
    """
    output_files: dict[str, str] = {}
    
    for pair in json_data.get("pairs", []):
        function_name = pair["function_name"]
        tests = pair["tests"]
        
        if format_style == "function":
            # Standalone test functions
            code = _generate_standalone_tests(function_name, tests)
            filename = f"test_{function_name}.py"
        elif format_style == "class":
            # Grouped in test class
            code = _generate_class_tests(function_name, tests)
            filename = f"test_{function_name}_class.py"
        elif format_style == "both":
            # Both formats
            standalone = _generate_standalone_tests(function_name, tests)
            grouped = _generate_class_tests(function_name, tests)
            output_files[f"test_{function_name}_standalone.py"] = standalone
            output_files[f"test_{function_name}_class.py"] = grouped
            continue
        else:
            raise ValueError(f"Unknown format_style: {format_style}")
        
        output_files[filename] = code
    
    return output_files


def _generate_standalone_tests(function_name: str, tests: list[dict[str, str]]) -> str:
    """Generate standalone test functions."""
    lines = [
        '"""Auto-generated test module."""',
        "",
        "# TODO: Update imports based on your project structure",
        "# from src.app.math_utils import add, subtract, divide",
        "",
    ]
    
    for test in tests:
        test_code = test["test_code"]
        # Convert method to function (remove self parameter, fix indentation)
        test_func = _method_to_function(test_code)
        lines.append(test_func)
        lines.append("")
    
    return "\n".join(lines)


def _generate_class_tests(function_name: str, tests: list[dict[str, str]]) -> str:
    """Generate tests grouped in a class."""
    class_name = f"Test{function_name.capitalize()}"
    
    lines = [
        '"""Auto-generated test module."""',
        "",
        "# TODO: Update imports",
        "# from src.app.math_utils import add, subtract, divide",
        "",
        f"class {class_name}:",
        '    """Test suite for ' + function_name + '."""',
        "",
    ]
    
    if not tests:
        lines.append("    pass")
    else:
        for test in tests:
            test_code = test["test_code"]
            # Keep method format but ensure proper indentation
            test_lines = test_code.split("\n")
            for line in test_lines:
                if line:
                    lines.append("    " + line)
                else:
                    lines.append("")
            lines.append("")
    
    return "\n".join(lines)


def _method_to_function(method_code: str) -> str:
    """Convert a test method to a standalone function."""
    lines = method_code.split("\n")
    result = []
    
    for i, line in enumerate(lines):
        if i == 0 and "def " in line and "(self)" in line:
            # Replace def method_name(self): with def method_name():
            new_line = line.replace("(self)", "()").replace("(self, ", "(")
            result.append(new_line)
        else:
            # Remove extra indentation from method body
            if line.startswith("    ") and i > 0:
                result.append(line[4:])
            else:
                result.append(line)
    
    return "\n".join(result)


def write_python_files(
    json_file: Path | str,
    output_dir: Path | str,
    format_style: str = "function",
) -> dict[str, str]:
    """
    Convert JSON to Python files and write them.
    
    Args:
        json_file: Path to function_test_pairs.json
        output_dir: Directory to write Python files to
        format_style: "function", "class", or "both"
    
    Returns:
        Dict of written file paths and their content
    """
    json_path = Path(json_file)
    output_path = Path(output_dir)
    
    # Read JSON
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    
    # Convert to Python
    python_files = convert_json_to_python(json_data, format_style)
    
    # Write files
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: dict[str, str] = {}
    
    for filename, code in python_files.items():
        file_path = output_path / filename
        file_path.write_text(code, encoding="utf-8")
        written_files[str(file_path)] = code
        print(f"[OK] Generated: {file_path}")
    
    return written_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python json_to_python.py <input.json> <output_dir> [format]")
        print("  format: function (default), class, or both")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2]
    format_style = sys.argv[3] if len(sys.argv) > 3 else "function"
    
    write_python_files(json_file, output_dir, format_style)
    print(f"\n[OK] Python files written to {output_dir}")
