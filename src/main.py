"""
Orchestrator — run from anywhere:

    python src/main.py /path/to/project

Optimizes Python functions in-place within the given project directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these values
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent

CONFIG = {
    # Where the spec-logic output JSON is written.
    "output": SRC_DIR / "spec_logic/output/results.json",

    # Where the optimizer output JSON is written.
    "optimizer_output": SRC_DIR / "optimizer_logic/output/results.json",

    # Minimum per-function line coverage required before we stop (0–100).
    "coverage_threshold": 80.0,

    # How many times Claude may attempt to improve tests for a single function.
    "max_iterations": 3,
}

# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable without pip install
# ---------------------------------------------------------------------------

sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv                          # noqa: E402
load_dotenv(SRC_DIR / "../.env")                        # repo-root .env
load_dotenv(SRC_DIR / ".env")                           # src/.env (optional override)

from spec_logic.langgraph_workflow import run_workflow           # noqa: E402
from optimizer_logic.optimizer import optimize_function         # noqa: E402
from optimizer_logic.function_spec import FunctionSpec          # noqa: E402
from convertor.inplace_rewriter import rewrite_functions_inplace  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(project_path: Path) -> list[dict]:
    """Run the full optimization pipeline (Phase 1 + 2 + 3) on *project_path*.

    Returns the list of optimizer result dicts.
    """
    # ── Phase 1: Spec-logic (analyse & generate tests) ───────────────────
    output = run_workflow(
        project_root=project_path,
        output_path=Path(CONFIG["output"]),
        coverage_threshold=float(CONFIG["coverage_threshold"]),
        max_iterations=int(CONFIG["max_iterations"]),
    )

    functions = output.get("functions", [])
    failed = [f for f in functions if f["status"] == "failed"]

    if failed:
        names = [f["id"] for f in failed]
        print(f"\n[pipeline] failed functions: {names}")

    # ── Phase 2: Optimize ────────────────────────────────────────────────
    optimizer_results = []
    for func_result in functions:
        if func_result["status"] not in ("passed_existing", "generated"):
            continue
        if not func_result["function_code"] or not func_result["test_code"]:
            continue

        spec = FunctionSpec(
            function_name=func_result["name"],
            module_path=func_result["file"],
            function_source=func_result["function_code"],
            test_source=func_result["test_code"],
        )
        result = optimize_function(spec)
        optimizer_results.append({
            "id": func_result["id"],
            "name": func_result["name"],
            "file": func_result["file"],
            **result,
        })

    optimizer_output_path = Path(CONFIG["optimizer_output"])
    optimizer_output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_output_path.write_text(
        json.dumps({
            "project_root": str(project_path),
            "generated_at": datetime.now().isoformat(),
            "functions": optimizer_results,
        }, indent=2)
    )

    # ── Phase 3: Rewrite in-place ────────────────────────────────────────
    modified = rewrite_functions_inplace(project_path, optimizer_results)
    print(f"\n[rewriter] {len(modified)} file(s) updated in {project_path}")

    return optimizer_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    project_path = Path(args.project_dir).resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"[ERROR] project_path does not exist: {project_path}")
        sys.exit(1)

    run_pipeline(project_path)
