"""CLI for converting function-test pair JSON to Python test files."""

import argparse
import sys
from pathlib import Path

from .json_to_python import write_python_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert function-test pairs JSON to Python test files"
    )
    parser.add_argument(
        "input_json",
        help="Path to function_test_pairs.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./generated_tests",
        help="Output directory for generated Python files (default: ./generated_tests)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["function", "class", "both"],
        default="function",
        help="Output format style (default: function)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.name.endswith(".json"):
        print("[ERROR] Input must be a .json file")
        sys.exit(1)
    
    try:
        print(f"[*] Converting {input_path.name} to Python files...")
        write_python_files(
            input_path,
            args.output,
            format_style=args.format,
        )
        print(f"[OK] Complete! Generated tests in: {args.output}")
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
