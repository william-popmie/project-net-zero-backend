"""
Orchestrator — run from anywhere:

    python src/main.py

Edit the CONFIG block below to change behaviour.
"""

from __future__ import annotations

import json
import subprocess
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
        print(f"[setup] Installing {len(valid_reqs)} packages from {req_file}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *valid_reqs],
            check=False,
        )


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
    input_repo_dir = Path(CONFIG["project_path"]).resolve()
    if not input_repo_dir.exists() or not input_repo_dir.is_dir():
        print(f"[ERROR] project_path does not exist: {input_repo_dir}")
        sys.exit(1)

    project_path = find_python_project_root(input_repo_dir)
    install_input_repo_requirements(project_path)

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

    # ── Phase 3: Convert ─────────────────────────────────────────────────────
    convertor_output_path = Path(CONFIG["convertor_output"]).resolve()
    written = write_python_files(optimizer_output_path, convertor_output_path)
    print(f"\n[convertor] {len(written)} file(s) written to {convertor_output_path}")
