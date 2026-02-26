"""
optimizer.py — entry point for the optimization phase.

Usage (standalone):
    python src/optimizer_logic/optimizer.py

Or from main.py:
    from optimizer_logic.optimizer import run_optimizer
    result = run_optimizer(spec_results_path=..., output_path=...)
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Maximum seconds to spend on a single function (API call + N parallel test runs + measurement).
# Increased from 17 to accommodate N parallel versions running concurrently.
FUNCTION_TIMEOUT_SECONDS = 60

# Number of parallel optimized versions to generate and test per Claude API call.
N_VERSIONS = 3

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
        from .graph import optimize_function_parallel
        from .venv_runner import create_shared_venv
    except ImportError:
        from optimizer_logic.graph import optimize_function_parallel  # type: ignore[no-redef]
        from optimizer_logic.venv_runner import create_shared_venv  # type: ignore[no-redef]

    # ── Resolve paths ─────────────────────────────────────────────────────────
    src_dir = Path(__file__).parent.parent  # src/
    if spec_results_path is None:
        spec_results_path = src_dir / "spec_logic" / "output" / "results.json"
    if output_path is None:
        output_path = src_dir / "optimizer_logic" / "output" / "result.json"

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
        return _write_output(output_path, project_root, skipped, summary=True)

    # ── Create shared venv (one per optimizer run) ─────────────────────────────
    with tempfile.TemporaryDirectory(prefix="optimizer_venv_") as venv_tmp:
        venv_dir = Path(venv_tmp) / "venv"
        _, python = create_shared_venv(project_root, venv_dir)

        all_results: list[dict] = list(skipped)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(to_optimize)
        for i, func in enumerate(to_optimize):
            print(f"\n[{i+1}/{total}] {func['id']}  ({func['file']})")

            source_file = project_root_path / func["file"]
            full_source = source_file.read_text()

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _future = _pool.submit(
                        optimize_function_parallel,
                        func_record=func,
                        project_root=project_root,
                        python_bin=python,
                        full_source=full_source,
                        n_versions=N_VERSIONS,
                        max_retries=2,
                    )
                    try:
                        final_result = _future.result(timeout=FUNCTION_TIMEOUT_SECONDS)
                    except concurrent.futures.TimeoutError:
                        print(f"  TIMEOUT after {FUNCTION_TIMEOUT_SECONDS}s — skipping")
                        _future.cancel()
                        all_results.append(_make_result(func, skip_reason="timeout"))
                        _write_output(output_path, project_root, all_results)
                        continue
            except Exception as exc:
                print(f"  error: {exc}")
                all_results.append(_make_result(func, skip_reason=f"error: {exc}"))
            else:
                all_results.append(_state_to_result(func, final_result))

            _write_output(output_path, project_root, all_results)

    return _write_output(output_path, project_root, all_results, summary=True)


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
        "optimized_function_code": func.get("function_code", ""),
        "optimized_test_code": func.get("spec_code", ""),
        "baseline_emissions_kg": None,
        "optimized_emissions_kg": None,
        "reduction_pct": 0.0,
        "success": False,
        "optimization_attempts": 0,
        "skip_reason": skip_reason,
    }


def _state_to_result(func: dict, state: dict) -> dict:
    success = state.get("success", False)
    baseline = state.get("baseline_emissions") or 0.0
    optimized_emissions = state.get("optimized_emissions")

    if success and optimized_emissions is not None and baseline > 0:
        reduction_pct = (baseline - optimized_emissions) / baseline * 100
    else:
        reduction_pct = 0.0

    return {
        "id": func.get("id", ""),
        "name": func.get("name", ""),
        "qualified_name": func.get("qualified_name", ""),
        "file": func.get("file", ""),
        "function_code": func.get("function_code", ""),
        "spec_code": func.get("spec_code", ""),
        "optimized_function_code": state.get("current_function_code") if success else func.get("function_code", ""),
        "optimized_test_code": func.get("spec_code", ""),
        "baseline_emissions_kg": state.get("baseline_emissions"),
        "optimized_emissions_kg": optimized_emissions if success else None,
        "reduction_pct": reduction_pct,
        "success": success,
        "optimization_attempts": state.get("attempt", 0),
        "skip_reason": state.get("skip_reason"),
    }


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so output is valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _write_output(
    output_path: Path,
    project_root: str,
    functions: list[dict],
    *,
    summary: bool = False,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "project_root": project_root,
        "generated_at": datetime.now().isoformat(),
        "functions": functions,
    }
    output_path.write_text(json.dumps(_sanitize(result), indent=2))
    if summary:
        successful = sum(1 for f in functions if f.get("success"))
        print(f"\nResults written to {output_path}")
        print(f"{successful}/{len(functions)} functions optimized successfully")
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
