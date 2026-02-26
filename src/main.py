"""
Orchestrator — run from anywhere:

    python src/main.py /path/to/project

Optimizes Python functions in-place within the given project directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these values
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent

CONFIG = {
    # Where the spec-logic output JSON is written.
    "output": SRC_DIR / "spec_logic/output/results.json",

    # Where the optimizer output JSON is written.
    "optimizer_output": SRC_DIR / "optimizer_logic/output/result.json",

    # Minimum per-function line coverage required before we stop (0–100).
    "coverage_threshold": 80.0,

    # How many times Claude may attempt to improve tests for a single function.
    "max_iterations": 3,
}

# ---------------------------------------------------------------------------
# Helpers: dynamic project-root discovery + dependency installation
# ---------------------------------------------------------------------------

def find_python_project_root(input_repo_dir: Path) -> Path:
    """Dynamically discover the actual Python project root inside input-repo/."""
    # If .py files exist directly in input-repo/, use it as root
    if any(input_repo_dir.glob("*.py")):
        return input_repo_dir
    # Otherwise look for first non-hidden subdirectory with Python files
    # Prefer the one with a requirements.txt (that's the project root)
    candidates = [
        d for d in sorted(input_repo_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    for subdir in candidates:
        if (subdir / "requirements.txt").exists():
            return subdir
    for subdir in candidates:
        if any(subdir.rglob("*.py")):
            return subdir
    return input_repo_dir  # fallback


def install_input_repo_requirements(project_root: Path) -> None:
    """Find and install requirements.txt from the project root or its parent."""
    req_file = None
    for candidate in [project_root / "requirements.txt",
                      project_root.parent / "requirements.txt"]:
        if candidate.exists():
            req_file = candidate
            break
    if not req_file:
        return

    valid_reqs = []
    for line in req_file.read_text().splitlines():
        line = line.split("#")[0].strip()  # strip inline comments
        if not line or line.startswith("pip "):
            continue
        valid_reqs.append(line)

    if valid_reqs:
        print(f"[setup] Installing {len(valid_reqs)} packages: {', '.join(valid_reqs)}", flush=True)
        print("[setup] (this may take a minute for large packages like numpy/pandas)...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *valid_reqs],
            check=False,
        )
        print("[setup] Packages installed.", flush=True)


# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable without pip install
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)  # flush every line immediately
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv                          # noqa: E402
load_dotenv(SRC_DIR / "../.env")                        # repo-root .env
load_dotenv(SRC_DIR / ".env")                           # src/.env (optional override)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(project_path: Path) -> list[dict]:
    """Run the full optimization pipeline (Phase 1 + 2 + 3) on *project_path*."""
    print("[setup] Loading pipeline modules (LangGraph, Anthropic, etc.)...", flush=True)
    from spec_logic.langgraph_workflow import run_workflow
    from convertor.inplace_rewriter import rewrite_functions_inplace
    from optimizer_logic.optimizer import run_optimizer
    from convertor.json_to_python import write_python_files
    print("[setup] Modules loaded.", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  PROJECT NET-ZERO — optimization pipeline")
    print(f"  Project: {project_path}")
    print(f"{'='*60}\n", flush=True)

    # ── Phase 1: Spec-logic (parse project + generate test specs) ────────
    print("┌─────────────────────────────────────────────────────────", flush=True)
    print("│ Phase 1/3 — Parse project & generate test specs", flush=True)
    print("└─────────────────────────────────────────────────────────", flush=True)
    output = run_workflow(
        project_root=project_path,
        output_path=Path(CONFIG["output"]),
    )

    functions = output.get("functions", [])
    generated = [f for f in functions if f.get("status") == "generated"]
    failed = [f for f in functions if f.get("status") == "failed"]

    print(f"\n✓ Phase 1 complete — {len(generated)}/{len(functions)} specs generated, {len(failed)} failed", flush=True)
    if failed:
        print("  Skipped (no spec):")
        for f in failed:
            print(f"    - {f['id']}")

    # ── Phase 2: Optimize ────────────────────────────────────────────────
    print(f"\n┌─────────────────────────────────────────────────────────", flush=True)
    print(f"│ Phase 2/3 — Optimize {len(generated)} function(s) with Claude", flush=True)
    print(f"└─────────────────────────────────────────────────────────", flush=True)
    optimizer_output_path = Path(CONFIG["optimizer_output"])
    run_optimizer(
        spec_results_path=Path(CONFIG["output"]),
        output_path=optimizer_output_path,
    )
    import json as _json
    optimizer_results = _json.loads(optimizer_output_path.read_text()).get("functions", [])
    successful = sum(1 for f in optimizer_results if f.get("success"))
    print(f"\n✓ Phase 2 complete — {successful}/{len(optimizer_results)} functions successfully optimized", flush=True)

    # ── Phase 3: Write output-repo (copy input + splice optimized) ───────
    print(f"\n┌─────────────────────────────────────────────────────────", flush=True)
    print(f"│ Phase 3/3 — Build output-repo with optimized source files", flush=True)
    print(f"└─────────────────────────────────────────────────────────", flush=True)
    input_repo_dir = Path(__file__).parent.parent / "input-repo"
    output_repo_dir = Path(__file__).parent.parent / "output-repo"
    written = write_python_files(
        json_file=optimizer_output_path,
        output_dir=output_repo_dir,
        input_repo_dir=input_repo_dir,
    )
    print(f"\n✓ Phase 3 complete — {len(written)} file(s) written to {output_repo_dir}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Pipeline complete!")
    print(f"  {successful}/{len(optimizer_results)} functions optimized")
    print(f"  Results: {optimizer_output_path}")
    print(f"{'='*60}\n", flush=True)

    return optimizer_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Start of main")
    parser = argparse.ArgumentParser(
        description="Optimize Python functions in a project directory."
    )

    parser.add_argument(
        "project_dir",
        nargs="?",
        default=str(SRC_DIR / "../input-repo"),
        help="Path to the project directory to optimize (default: input-repo/)",
    )
    args = parser.parse_args()

    print("arguments parsed")

    input_repo_dir = Path(args.project_dir).resolve()
    if not input_repo_dir.exists() or not input_repo_dir.is_dir():
        print(f"[ERROR] project_path does not exist: {input_repo_dir}")
        sys.exit(1)

    print(f"[setup] Input dir:   {input_repo_dir}")
    project_path = find_python_project_root(input_repo_dir)
    print(f"[setup] Project root: {project_path}")
    install_input_repo_requirements(project_path)

    run_pipeline(project_path)
