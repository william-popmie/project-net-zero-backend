#!/usr/bin/env python3
"""Helper script to test the parser → spec-generation pipeline in isolation.

Usage:
    python src/test_pipeline.py
    python src/test_pipeline.py --input path/to/repo
    python src/test_pipeline.py --input path/to/repo --output path/to/out.json
    python src/test_pipeline.py --parse-only   # just print discovered functions, no API calls
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(SRC_DIR / "../.env")
load_dotenv(SRC_DIR / ".env")

from parser.graph_parser import collect_functions, find_requirements
from spec_logic.langgraph_workflow import run_workflow

DEFAULT_INPUT = SRC_DIR / "../input-repo"
DEFAULT_OUTPUT = SRC_DIR / "spec_logic/output/results.json"


def find_project_root(input_repo: Path) -> Path:
    if any(input_repo.glob("*.py")):
        return input_repo
    candidates = [d for d in sorted(input_repo.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    for d in candidates:
        if (d / "requirements.txt").exists():
            return d
    for d in candidates:
        if any(d.rglob("*.py")):
            return d
    return input_repo


def install_requirements(project_root: Path) -> None:
    req = find_requirements(project_root)
    if req is None:
        return
    lines = [ln.split("#")[0].strip() for ln in req.read_text().splitlines()]
    reqs = [r for r in lines if r and not r.startswith("pip ")]
    if reqs:
        print(f"[setup] Installing {len(reqs)} packages from {req}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], check=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Parser → spec-gen pipeline tester")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input-repo or project root")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write results.json")
    p.add_argument("--parse-only", action="store_true", help="Only run the parser, skip Claude API calls")
    args = p.parse_args()

    input_repo = args.input.resolve()
    if not input_repo.exists():
        print(f"[ERROR] Input not found: {input_repo}")
        sys.exit(1)

    project_root = find_project_root(input_repo)
    print(f"[setup] Project root : {project_root}")
    install_requirements(project_root)

    if args.parse_only:
        print(f"\nScanning project: {project_root}")
        functions = collect_functions(project_root)
        print(f"\nFound {len(functions)} source functions:")
        for fn in functions:
            print(f"  {fn.id}  (lines {fn.line}–{fn.end_line})")
        return

    run_workflow(project_root=project_root, output_path=args.output.resolve())


if __name__ == "__main__":
    main()
