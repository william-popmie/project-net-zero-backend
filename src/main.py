"""
Orchestrator — run from anywhere:

    python src/main.py

Edit the CONFIG block below to change behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these values
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent

CONFIG = {
    # Path to the Python project to analyse.
    "project_path": SRC_DIR / "../input-repo",

    # Where the output JSON is written.
    "output": SRC_DIR / "spec_logic/output/results.json",

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

from spec_logic.langgraph_workflow import run_workflow   # noqa: E402

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

    if failed:
        print("\nFailed functions:")
        for f in failed:
            print(f"  - {f['id']}")
            for err in f.get("errors", []):
                print(f"      {err}")
        sys.exit(1)
