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
import tempfile
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
    """Run the full optimization pipeline (Phase 1 + 2 + 3) on *project_path*.

    Phase 1 (Gemini spec gen) and Phase 2 (Claude optimization) run as a
    parallel producer-consumer pipeline: Gemini generates specs for function N+1
    while Claude is already optimizing function N.
    """
    print("[setup] Loading pipeline modules...", flush=True)
    import concurrent.futures
    from parser.graph_parser import collect_functions
    from spec_logic.gemini_spec_generator import generate_spec as gemini_generate_spec
    from optimizer_logic.venv_runner import create_shared_venv
    from optimizer_logic.graph import optimize_function_parallel
    from optimizer_logic.optimizer import (
        _make_result, _state_to_result, _write_output,
        FUNCTION_TIMEOUT_SECONDS, N_VERSIONS,
    )
    from convertor.json_to_python import write_python_files
    print("[setup] Modules loaded.", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  PROJECT NET-ZERO — optimization pipeline")
    print(f"  Project: {project_path}")
    print(f"{'='*60}\n", flush=True)

    GEMINI_WORKERS = 5
    CLAUDE_WORKERS = 1

    project_root = project_path.resolve()
    project_root_str = str(project_root)
    spec_output_path = Path(CONFIG["output"])
    optimizer_output_path = Path(CONFIG["optimizer_output"])

    # ── Phase 1 + 2: Parallel producer-consumer pipeline ─────────────────
    print("┌─────────────────────────────────────────────────────────", flush=True)
    print("│ Phase 1+2/3 — Gemini spec gen → Claude optimization (parallel)", flush=True)
    print("└─────────────────────────────────────────────────────────", flush=True)

    print(f"\nScanning project: {project_root}", flush=True)
    functions_info = collect_functions(project_root)
    print(f"\nFound {len(functions_info)} source functions\n", flush=True)

    # ── Inner helpers (closures over imported names) ───────────────────────

    def _generate_spec_task(func, project_root):
        """Generate spec for one function via Gemini. Never raises."""
        func_record = {
            "id": func.id,
            "name": func.name,
            "qualified_name": func.qualified_name,
            "file": func.file_path,
            "line": func.line,
            "end_line": func.end_line,
            "function_code": func.source_code,
            "spec_code": "",
            "status": "pending",
            "error": "",
        }
        try:
            print(f"  [gemini] generating spec for {func.id} ...", flush=True)
            code = gemini_generate_spec(func.id, func.source_code, func.file_path)
            func_record["spec_code"] = code
            func_record["status"] = "generated"
            print(f"  [gemini] OK  {func.id} ({len(code.splitlines())} lines)", flush=True)
        except Exception as e:
            func_record["status"] = "failed"
            func_record["error"] = str(e)
            print(f"  [gemini] FAILED {func.id}: {e}", flush=True)
        return func_record

    def _optimize_task(func_record, python_bin, project_root):
        """Optimize one function via Claude. Returns None on timeout/error."""
        func_id = func_record["id"]
        source_file = Path(project_root) / func_record["file"]
        try:
            full_source = source_file.read_text()
        except Exception as e:
            print(f"  [optimizer] ERROR reading {source_file}: {e}", flush=True)
            return None

        print(f"\n[optimizer] {func_id}", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                _future = _pool.submit(
                    optimize_function_parallel,
                    func_record=func_record,
                    project_root=project_root,
                    python_bin=python_bin,
                    full_source=full_source,
                    n_versions=N_VERSIONS,
                    max_retries=2,
                )
                try:
                    return _future.result(timeout=FUNCTION_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    print(f"  TIMEOUT after {FUNCTION_TIMEOUT_SECONDS}s — {func_id}", flush=True)
                    _future.cancel()
                    return None
        except Exception as e:
            print(f"  [optimizer] ERROR {func_id}: {e}", flush=True)
            return None

    # ── Pipeline execution ─────────────────────────────────────────────────

    all_results: list[dict] = []
    all_spec_results: list[dict] = []

    spec_output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="optimizer_venv_") as venv_tmp_str:
        venv_dir = Path(venv_tmp_str) / "venv"
        _, python_bin = create_shared_venv(project_root_str, venv_dir)

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as gemini_pool,
            concurrent.futures.ThreadPoolExecutor(max_workers=CLAUDE_WORKERS) as opt_pool,
        ):
            # Submit all spec tasks immediately
            spec_futures = {
                gemini_pool.submit(_generate_spec_task, func, project_root): func
                for func in functions_info
            }

            opt_futures: dict = {}

            # As specs complete, feed successful ones into the optimizer pool
            for spec_future in concurrent.futures.as_completed(spec_futures):
                func_record = spec_future.result()
                all_spec_results.append(func_record)

                if func_record["status"] == "generated":
                    opt_future = opt_pool.submit(
                        _optimize_task, func_record, python_bin, project_root_str
                    )
                    opt_futures[opt_future] = func_record
                else:
                    all_results.append(_make_result(func_record, skip_reason="no_spec"))
                    _write_output(optimizer_output_path, project_root_str, all_results)

            # Write spec results.json once all specs are submitted
            spec_output = {
                "project_root": project_root_str,
                "generated_at": datetime.now().isoformat(),
                "functions": all_spec_results,
            }
            spec_output_path.write_text(
                json.dumps(spec_output, indent=2), encoding="utf-8"
            )
            generated_count = sum(1 for f in all_spec_results if f["status"] == "generated")
            failed_count = len(all_spec_results) - generated_count
            print(
                f"\n✓ Phase 1 complete — {generated_count}/{len(all_spec_results)} "
                f"specs generated, {failed_count} failed",
                flush=True,
            )
            if failed_count:
                print("  Skipped (no spec):")
                for f in all_spec_results:
                    if f["status"] != "generated":
                        print(f"    - {f['id']}")

            # Collect optimizer results as they complete
            print(
                f"\n┌─────────────────────────────────────────────────────────", flush=True
            )
            print(
                f"│ Collecting optimizer results for {len(opt_futures)} function(s)", flush=True
            )
            print(
                f"└─────────────────────────────────────────────────────────", flush=True
            )
            for opt_future in concurrent.futures.as_completed(opt_futures):
                func_record = opt_futures[opt_future]
                try:
                    final = opt_future.result()
                    if final is None:
                        all_results.append(_make_result(func_record, skip_reason="timeout"))
                    else:
                        all_results.append(_state_to_result(func_record, final))
                except Exception as e:
                    all_results.append(_make_result(func_record, skip_reason=f"error: {e}"))
                _write_output(optimizer_output_path, project_root_str, all_results)

    successful = sum(1 for f in all_results if f.get("success"))
    print(
        f"\n✓ Phase 2 complete — {successful}/{len(all_results)} "
        f"functions successfully optimized",
        flush=True,
    )

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
    print(f"  {successful}/{len(all_results)} functions optimized")
    print(f"  Results: {optimizer_output_path}")
    print(f"{'='*60}\n", flush=True)

    return all_results


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
