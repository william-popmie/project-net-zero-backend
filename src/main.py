"""
Orchestrator — run from anywhere:

    python src/main.py
    python src/main.py --model crusoe

Edit the CONFIG block below to change other behaviour.
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
    # Path to the Python project to analyse.
    "project_path": SRC_DIR / "../input-repo",

    # Where the spec-logic output JSON is written.
    "output": SRC_DIR / "spec_logic/output/results.json",

    # Where the optimizer output JSON is written.
    "optimizer_output": SRC_DIR / "optimizer_logic/output/results.json",

    # Where the convertor writes the final reconstructed Python files.
    "convertor_output": SRC_DIR / "../output-repo",

    # Minimum per-function line coverage required before we stop (0–100).
    "coverage_threshold": 80.0,

    # How many times the model may attempt to improve tests for a single function.
    "max_iterations": 3,

    # Inference engine: "claude" or "crusoe"
    "engine": "claude",
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
from convertor.json_to_python import write_python_files         # noqa: E402

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Net Zero optimiser")
    parser.add_argument(
        "--model",
        choices=["claude", "crusoe"],
        default=CONFIG["engine"],
        help="Inference engine to use (default: %(default)s)",
    )
    args = parser.parse_args()
    CONFIG["engine"] = args.model

    project_path = Path(CONFIG["project_path"]).resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"[ERROR] project_path does not exist: {project_path}")
        sys.exit(1)

    output = run_workflow(
        project_root=project_path,
        output_path=Path(CONFIG["output"]),
        coverage_threshold=float(CONFIG["coverage_threshold"]),
        max_iterations=int(CONFIG["max_iterations"]),
        engine=CONFIG["engine"],
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

    # ── Phase 2: Optimize ────────────────────────────────────────────────────
    optimizer_results = []
    for func_result in output.get("functions", []):
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
        result = optimize_function(spec, engine=CONFIG["engine"])
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

    # ── Phase 3: Convert ─────────────────────────────────────────────────────
    convertor_output_path = Path(CONFIG["convertor_output"]).resolve()
    written = write_python_files(optimizer_output_path, convertor_output_path)
    print(f"\n[convertor] {len(written)} file(s) written to {convertor_output_path}")
