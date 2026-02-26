"""
graph.py — Parallel optimization pipeline (no LangGraph).

Flow per function:
  1. Baseline measurement starts in a background thread
  2. Generate N optimized versions in a single Claude API call (sequential, rate-limited)
  3. Wait for baseline; skip if failed
  4. Test all N versions IN PARALLEL (spec + emissions per version, cross-cancellation)
  5. Pick best passing version; retry with feedback on failure (up to max_retries)
"""

from __future__ import annotations

import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from .venv_runner import (
    build_project_dir,
    measure_emissions_via_pytest,
    measure_emissions_via_pytest_cancellable,
    replace_function_in_source,
    run_spec_cancellable,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Indentation helpers (preserved from original graph.py)
# ---------------------------------------------------------------------------

def _extract_indented_function(full_source: str, start_line: int, end_line: int) -> str:
    """Return the lines [start_line..end_line] (1-indexed, inclusive) from full_source."""
    lines = full_source.splitlines()
    return "\n".join(lines[start_line - 1 : end_line])


def _get_indent(full_source: str, start_line: int) -> str:
    """Return the leading whitespace of the function definition line."""
    lines = full_source.splitlines()
    if start_line - 1 < len(lines):
        line = lines[start_line - 1]
        return line[: len(line) - len(line.lstrip())]
    return ""


def _apply_indent(code: str, indent: str) -> str:
    """Re-apply *indent* to every line of *code* if the first line lacks it."""
    if not indent:
        return code
    lines = code.splitlines()
    if not lines:
        return code
    if lines[0].startswith(indent):
        return code  # already correctly indented
    return "\n".join(indent + line for line in lines)


# ---------------------------------------------------------------------------
# Claude API: generate N versions in one call
# ---------------------------------------------------------------------------

def generate_n_versions(
    func_record: dict,
    full_source: str,
    n: int,
    attempt: int,
    feedback: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Call Claude once asking for N optimized versions.

    Returns a list of (version_code, full_source_with_version) tuples.
    May return fewer than N if Claude doesn't provide enough code blocks.
    """
    start_line = func_record.get("line", 1)
    end_line = func_record.get("end_line", start_line)
    indent = _get_indent(full_source, start_line)
    current_code = _extract_indented_function(full_source, start_line, end_line)
    spec_code = func_record.get("spec_code", "")

    print(f"  [generate_n_versions] attempt {attempt}, requesting {n} versions")

    user_content = (
        f"Optimize the following Python function to reduce CPU usage and energy consumption.\n\n"
        f"IMPORTANT rules:\n"
        f"1. Return EXACTLY {n} different optimized versions.\n"
        f"2. Each version must be in its own separate ```python code block.\n"
        f"3. Preserve the EXACT leading indentation of every line "
        f"(the function may be a class method).\n"
        f"4. Do not add any explanation outside the code blocks.\n\n"
        f"Original function (keep the same indentation):\n```python\n{current_code}\n```"
    )

    if attempt > 1 and feedback:
        user_content += (
            f"\n\nPrevious attempt failed. Failure details:\n{feedback}\n\n"
            f"The function must pass these tests:\n```python\n{spec_code}\n```"
        )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=(
            "You are an expert Python performance engineer. "
            f"When asked to optimize a function, return EXACTLY {n} different optimized versions, "
            "each in its own separate ```python code block. "
            "Match the original indentation exactly — if the original has 4-space leading indent, keep it. "
            "No explanations, no other text."
        ),
        messages=[{"role": "user", "content": user_content}],
    )

    raw = message.content[0].text
    blocks = re.findall(r"```python\s*(.*?)```", raw, re.DOTALL)

    versions: list[tuple[str, str]] = []
    for block in blocks[:n]:
        code = block.rstrip("\n")
        code = _apply_indent(code, indent)
        new_full_source = replace_function_in_source(full_source, code, start_line, end_line)
        versions.append((code, new_full_source))

    print(f"  [generate_n_versions] got {len(versions)} versions")
    return versions


# ---------------------------------------------------------------------------
# Per-version parallel testing with cross-cancellation
# ---------------------------------------------------------------------------

def test_single_version(
    version_code: str,
    version_full_source: str,
    func_record: dict,
    python_bin: Path,
    baseline_emissions: float,
) -> dict:
    """Test one optimized version: spec tests + emissions in parallel with cross-cancellation.

    Returns dict with keys: code, passed_tests, test_output, emissions, passed_emissions.
    """
    cancel_spec = threading.Event()
    cancel_carbon = threading.Event()

    with tempfile.TemporaryDirectory(prefix="optimizer_version_") as tmp_str:
        tmp = Path(tmp_str)
        build_project_dir(tmp, func_record, version_full_source)

        def run_spec_thread() -> tuple[bool, str]:
            passed, output = run_spec_cancellable(python_bin, tmp, cancel_spec)
            if not passed:
                cancel_carbon.set()
            return passed, output

        def run_carbon_thread() -> tuple[float, bool]:
            emissions, tests_passed = measure_emissions_via_pytest_cancellable(
                python_bin, tmp, cancel_carbon
            )
            # Only cross-cancel spec if carbon completely failed (couldn't measure)
            if emissions == 0.0 and not tests_passed:
                cancel_spec.set()
            return emissions, tests_passed

        with ThreadPoolExecutor(max_workers=2) as executor:
            spec_future = executor.submit(run_spec_thread)
            carbon_future = executor.submit(run_carbon_thread)
            spec_passed, test_output = spec_future.result()
            emissions, _ = carbon_future.result()

    passed_emissions = baseline_emissions > 0 and emissions < baseline_emissions

    return {
        "code": version_code,
        "passed_tests": spec_passed,
        "test_output": test_output,
        "emissions": emissions,
        "passed_emissions": passed_emissions,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def optimize_function_parallel(
    func_record: dict,
    project_root: str,
    python_bin: Path,
    full_source: str,
    n_versions: int = 3,
    max_retries: int = 2,
) -> dict:
    """Parallel optimizer replacing build_graph().invoke(initial_state).

    Returns a dict compatible with _state_to_result() in optimizer.py:
      success, baseline_emissions, optimized_emissions,
      current_function_code, attempt, skip_reason.
    """
    func_id = func_record.get("id", func_record.get("name", "?"))
    start_line = func_record.get("line", 1)
    end_line = func_record.get("end_line", start_line)
    original_code = _extract_indented_function(full_source, start_line, end_line)

    print(f"[optimize_function_parallel] {func_id} — baseline + {n_versions} versions")

    # ── Step 1 & 2: Baseline in background; generate versions concurrently ────
    with ThreadPoolExecutor(max_workers=1) as baseline_pool:
        def _measure_baseline() -> tuple[float, bool]:
            with tempfile.TemporaryDirectory(prefix="optimizer_baseline_") as tmp_str:
                tmp = Path(tmp_str)
                build_project_dir(tmp, func_record, full_source)
                return measure_emissions_via_pytest(python_bin, tmp, runs=1)

        baseline_future = baseline_pool.submit(_measure_baseline)

        # While baseline runs, call Claude (rate-limited — sequential)
        versions = generate_n_versions(func_record, full_source, n_versions, attempt=1)

        # ── Step 3: Wait for baseline ──────────────────────────────────────────
        avg_emissions, all_passed = baseline_future.result()

    if not all_passed or avg_emissions == 0.0:
        reason = "tests failed" if not all_passed else "measurement returned 0"
        print(f"[optimize_function_parallel] SKIP {func_id} — baseline {reason}")
        return {
            "success": False,
            "baseline_emissions": 0.0,
            "optimized_emissions": None,
            "current_function_code": original_code,
            "attempt": 0,
            "skip_reason": "baseline_measurement_failed",
        }

    print(f"[optimize_function_parallel] baseline = {avg_emissions:.2e} kg CO2eq")
    baseline_emissions = avg_emissions

    # ── Step 4: Retry loop ─────────────────────────────────────────────────────
    feedback: Optional[str] = None
    attempt = 0

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            # Regenerate with failure feedback
            versions = generate_n_versions(
                func_record, full_source, n_versions, attempt=attempt, feedback=feedback
            )

        if not versions:
            print(f"  [attempt {attempt}] no versions generated")
            continue

        print(f"  [attempt {attempt}] testing {len(versions)} versions in parallel")

        # Test all versions in parallel
        with ThreadPoolExecutor(max_workers=len(versions)) as test_pool:
            test_futures = [
                test_pool.submit(
                    test_single_version,
                    version_code,
                    version_full_source,
                    func_record,
                    python_bin,
                    baseline_emissions,
                )
                for version_code, version_full_source in versions
            ]
            results = [f.result() for f in test_futures]

        # Pick best passing version
        passing = [r for r in results if r["passed_tests"] and r["passed_emissions"]]
        if passing:
            best = min(passing, key=lambda r: r["emissions"])
            reduction = (baseline_emissions - best["emissions"]) / baseline_emissions * 100
            print(
                f"  [attempt {attempt}] SUCCESS — "
                f"{baseline_emissions:.2e} → {best['emissions']:.2e} kg CO2eq "
                f"({reduction:.1f}% reduction)"
            )
            return {
                "success": True,
                "baseline_emissions": baseline_emissions,
                "optimized_emissions": best["emissions"],
                "current_function_code": best["code"],
                "attempt": attempt,
                "skip_reason": None,
            }

        # Build feedback string for next attempt
        feedback_parts: list[str] = []
        for i, r in enumerate(results):
            if not r["passed_tests"]:
                feedback_parts.append(
                    f"Version {i + 1}: FAILED spec tests.\n{r['test_output'][:500]}"
                )
            elif not r["passed_emissions"]:
                pct = (
                    (r["emissions"] - baseline_emissions) / baseline_emissions * 100
                    if baseline_emissions > 0
                    else 0.0
                )
                feedback_parts.append(
                    f"Version {i + 1}: passed tests but {pct:.1f}% worse emissions than baseline"
                )
        feedback = "\n\n".join(feedback_parts)
        print(f"  [attempt {attempt}] all {len(results)} versions failed, will retry")

    print(f"[optimize_function_parallel] NO IMPROVEMENT after {attempt} attempt(s) for {func_id}")
    return {
        "success": False,
        "baseline_emissions": baseline_emissions,
        "optimized_emissions": None,
        "current_function_code": original_code,
        "attempt": attempt,
        "skip_reason": None,
    }
