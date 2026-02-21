"""
Entry point — run this script from anywhere:

    python spec-logic/run.py

Edit the CONFIG block below to change behaviour.
No installation or CLI flags needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these values
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

CONFIG = {
    # Path to the Python project you want to analyse.
    "project_path": SCRIPT_DIR / "../input/sample_project",

    # Where the output JSON is written (relative to this script).
    "output": SCRIPT_DIR / "output/results.json",

    # Minimum per-function line coverage required before we stop (0–100).
    "coverage_threshold": 80.0,

    # How many times Claude may attempt to improve tests for a single function.
    "max_iterations": 3,
}

# ---------------------------------------------------------------------------
# Bootstrap: make the package importable without pip install
# ---------------------------------------------------------------------------

sys.path.insert(0, str(SCRIPT_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402 — must come after sys.path tweak
load_dotenv(SCRIPT_DIR / ".env")          # spec-logic/.env
load_dotenv(SCRIPT_DIR / "../.env")       # repo-root .env (fallback)

from function_spec_graph.langgraph_workflow import run_workflow  # noqa: E402

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_path = Path(CONFIG["project_path"]).resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"[ERROR] project_path does not exist: {project_path}")
        sys.exit(1)

    output = run_workflow(
        project_root=project_path,
        output_path=Path(CONFIG["output"]),
        coverage_threshold=float(CONFIG["coverage_threshold"]),
        max_iterations=int(CONFIG["max_iterations"]),
    )

    functions = output.get("functions", [])
    failed = [f for f in functions if f["status"] == "failed"]

    print(f"\nSummary: {len(functions)} functions processed")
    print(f"  passed_existing : {sum(1 for f in functions if f['status'] == 'passed_existing')}")
    print(f"  generated       : {sum(1 for f in functions if f['status'] == 'generated')}")
    print(f"  failed          : {len(failed)}")

    if failed:
        print("\nFailed functions:")
        for f in failed:
            print(f"  - {f['id']}")
            for err in f.get("errors", []):
                print(f"      {err}")
        sys.exit(1)
