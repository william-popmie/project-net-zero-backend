"""
optimizer.py — entry point for the optimization phase.

Usage (standalone):
    python src/optimizer_logic/optimizer.py

Or from main.py:
    from optimizer_logic.optimizer import run_optimizer
    result = run_optimizer(spec_results_path=..., output_path=...)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def run_optimizer(
    spec_results_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Read spec_logic/output/results.json, run per-function LangGraph, write output.

    Returns the full output dict (with "functions" list).
    """
    # Lazy imports — work whether called as package or standalone script
    try:
        from .graph import build_graph
        from .venv_runner import create_shared_venv
    except ImportError:
        from optimizer_logic.graph import build_graph  # type: ignore[no-redef]
        from optimizer_logic.venv_runner import create_shared_venv  # type: ignore[no-redef]

    # ── Resolve paths ─────────────────────────────────────────────────────────
    src_dir = Path(__file__).parent.parent  # src/
    if spec_results_path is None:
        spec_results_path = src_dir / "spec_logic" / "output" / "results.json"
    if output_path is None:
        output_path = src_dir / "optimizer_logic" / "output" / "results.json"

    # ── Load spec results ─────────────────────────────────────────────────────
    if not spec_results_path.exists():
        raise FileNotFoundError(f"Spec results not found: {spec_results_path}")

    spec_data = json.loads(spec_results_path.read_text())
    project_root: str = spec_data.get("project_root", "")
    functions: list[dict] = spec_data.get("functions", [])

    project_root_path = Path(project_root)

    # ── Filter functions ───────────────────────────────────────────────────────
    to_optimize: list[dict] = []
    skipped: list[dict] = []

    for func in functions:
        if func.get("status") != "generated" or not func.get("spec_code"):
            skipped.append(_make_result(func, skip_reason="no_spec"))
            continue

        source_file = project_root_path / func["file"]
        if not source_file.exists():
            skipped.append(_make_result(func, skip_reason="source_file_not_found"))
            continue

        to_optimize.append(func)

    if not to_optimize:
        print("[optimizer] No functions to optimize — all skipped.")
        result = _write_output(output_path, project_root, skipped)
        return result

    # ── Create shared venv (one per optimizer run) ─────────────────────────────
    with tempfile.TemporaryDirectory(prefix="optimizer_venv_") as venv_tmp:
        venv_dir = Path(venv_tmp) / "venv"
        _, python = create_shared_venv(project_root, venv_dir)

        graph = build_graph()
        opt_results: list[dict] = []

        for func in to_optimize:
            source_file = project_root_path / func["file"]
            full_source = source_file.read_text()

            print(f"\n[optimizer] Processing: {func['id']}")

            initial_state = {
                "func_record": func,
                "project_root": project_root,
                "python_bin": python,
                "full_source": full_source,
                "current_function_code": func["function_code"],
                "current_full_source": full_source,
                "baseline_emissions": 0.0,
                "optimized_emissions": None,
                "test_passed": False,
                "attempt": 0,
                "max_attempts": 2,
                "last_test_output": "",
                "success": False,
                "skip_reason": None,
            }

            try:
                final_state = graph.invoke(initial_state)
            except Exception as exc:
                print(f"[optimizer] ERROR for {func['id']}: {exc}")
                opt_results.append(_make_result(func, skip_reason=f"error: {exc}"))
                continue

            opt_results.append(_state_to_result(func, final_state))

        all_results = skipped + opt_results

    return _write_output(output_path, project_root, all_results)


# ---------------------------------------------------------------------------
# Backwards-compat shim (remove once callers are updated)
# ---------------------------------------------------------------------------

def optimize_function(spec) -> dict:  # noqa: ANN001
    raise NotImplementedError(
        "optimize_function() is removed. Use run_optimizer() instead."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(func: dict, *, skip_reason: str) -> dict:
    return {
        "id": func.get("id", ""),
        "name": func.get("name", ""),
        "qualified_name": func.get("qualified_name", ""),
        "file": func.get("file", ""),
        "function_code": func.get("function_code", ""),
        "spec_code": func.get("spec_code", ""),
        "baseline_emissions_kg": None,
        "optimized_source": None,
        "optimized_emissions_kg": None,
        "success": False,
        "optimization_attempts": 0,
        "skip_reason": skip_reason,
    }


def _state_to_result(func: dict, state: dict) -> dict:
    success = state.get("success", False)
    optimized_emissions = state.get("optimized_emissions")

    return {
        "id": func.get("id", ""),
        "name": func.get("name", ""),
        "qualified_name": func.get("qualified_name", ""),
        "file": func.get("file", ""),
        "function_code": func.get("function_code", ""),
        "spec_code": func.get("spec_code", ""),
        "baseline_emissions_kg": state.get("baseline_emissions"),
        "optimized_source": state.get("current_function_code") if success else None,
        "optimized_emissions_kg": optimized_emissions if success else None,
        "success": success,
        "optimization_attempts": state.get("attempt", 0),
        "skip_reason": state.get("skip_reason"),
    }


def _write_output(
    output_path: Path,
    project_root: str,
    functions: list[dict],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "project_root": project_root,
        "generated_at": datetime.now().isoformat(),
        "functions": functions,
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\n[optimizer] Results written to {output_path}")
    successful = sum(1 for f in functions if f.get("success"))
    print(f"[optimizer] {successful}/{len(functions)} functions optimized successfully")
    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).parent.parent.parent / ".env")
    run_optimizer()
