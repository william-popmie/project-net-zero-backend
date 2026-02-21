import os
import subprocess
import sys
import tempfile
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
from langgraph.graph import StateGraph, END

from .function_spec import FunctionSpec
from .emissions import measure_emissions_for_source
from . import llm_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OptimizerState(TypedDict):
    spec: FunctionSpec
    current_source: str
    baseline_emissions: float
    current_emissions: float
    test_passed: bool
    attempt: int
    max_attempts: int
    last_test_output: str
    success: bool
    engine: str  # "claude" or "crusoe"
    inference_duration: float
    inference_tokens: int


# ── Nodes ────────────────────────────────────────────────────────────────────

def measure_baseline(state: OptimizerState) -> OptimizerState:
    print("[measure_baseline] measuring baseline emissions...")
    spec = state["spec"]
    emissions = measure_emissions_for_source(spec.function_source, spec.function_name)
    print(f"[measure_baseline] baseline = {emissions:.2e} kg CO2eq")
    return {
        **state,
        "baseline_emissions": emissions,
        "current_source": spec.function_source,
    }


def optimize(state: OptimizerState) -> OptimizerState:
    attempt = state["attempt"] + 1
    print(f"[optimize] attempt {attempt}/{state['max_attempts']} (engine={state['engine']})")

    spec = state["spec"]
    last_test_output = state["last_test_output"] if attempt > 1 else ""

    result = llm_client.rewrite(
        source_code=state["current_source"],
        test_code=spec.test_source,
        last_test_output=last_test_output,
        engine=state["engine"],
    )

    optimized_source = result["rewritten_code"]
    print(f"[optimize] got optimized source ({len(optimized_source)} chars) in {result['duration_seconds']:.1f}s")
    return {
        **state,
        "current_source": optimized_source,
        "attempt": attempt,
        "inference_duration": state.get("inference_duration", 0) + result["duration_seconds"],
        "inference_tokens": state.get("inference_tokens", 0) + result["usage"].get("total_tokens", 0),
    }


def run_tests(state: OptimizerState) -> OptimizerState:
    print("[run_tests] running pytest...")
    spec = state["spec"]
    # Strip `self` from test function signatures — test_source may contain
    # class methods extracted without the class wrapper.
    clean_test = re.sub(r"def (test_\w+)\(self(?:,\s*)?", r"def \1(", spec.test_source)

    # Strip imports of the source module — the function is inlined in the combined
    # file so any "from src.app.math_utils import ..." line would fail when pytest
    # runs the temp file from /tmp/ where that package doesn't exist.
    module_dotted = Path(spec.module_path).with_suffix("").as_posix().replace("/", ".")
    clean_test = re.sub(
        r"^from\s+" + re.escape(module_dotted) + r"\s+import\s+[^\n]*\n?",
        "",
        clean_test,
        flags=re.MULTILINE,
    )

    combined = "import pytest\n\n" + state["current_source"] + "\n\n" + clean_test

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(combined)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", tmp_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        print(f"[run_tests] {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(output)
        return {**state, "test_passed": passed, "last_test_output": output}
    finally:
        os.unlink(tmp_path)


def measure_emissions(state: OptimizerState) -> OptimizerState:
    print("[measure_emissions] measuring optimized emissions...")
    spec = state["spec"]
    emissions = measure_emissions_for_source(state["current_source"], spec.function_name)
    print(f"[measure_emissions] current = {emissions:.2e} kg CO2eq")
    return {**state, "current_emissions": emissions}


def save_output(state: OptimizerState) -> OptimizerState:
    baseline = state["baseline_emissions"]
    current = state["current_emissions"]

    # If no improvement was achieved, fall back to the original source.
    if current >= baseline:
        print(f"[save_output] no improvement after {state['attempt']} attempt(s) — keeping original")
        return {
            **state,
            "current_source": state["spec"].function_source,
            "success": True,
        }

    reduction = (baseline - current) / baseline * 100 if baseline > 0 else 0.0
    print(f"[save_output] baseline:  {baseline:.2e} kg CO2eq")
    print(f"[save_output] optimized: {current:.2e} kg CO2eq")
    print(f"[save_output] reduction: {reduction:.1f}%")

    return {**state, "success": True}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_tests(state: OptimizerState) -> str:
    if state["test_passed"]:
        return "measure_emissions"
    if state["attempt"] < state["max_attempts"]:
        return "optimize"
    return END


def route_after_emissions(state: OptimizerState) -> str:
    if state["current_emissions"] < state["baseline_emissions"]:
        return "save_output"
    if state["attempt"] < state["max_attempts"]:
        return "optimize"
    return "save_output"  # exhausted attempts — save_output will restore original


# ── Builder ───────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(OptimizerState)

    builder.add_node("measure_baseline", measure_baseline)
    builder.add_node("optimize", optimize)
    builder.add_node("run_tests", run_tests)
    builder.add_node("measure_emissions", measure_emissions)
    builder.add_node("save_output", save_output)

    builder.set_entry_point("measure_baseline")
    builder.add_edge("measure_baseline", "optimize")
    builder.add_edge("optimize", "run_tests")
    builder.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {"measure_emissions": "measure_emissions", "optimize": "optimize", END: END},
    )
    builder.add_conditional_edges(
        "measure_emissions",
        route_after_emissions,
        {"save_output": "save_output", "optimize": "optimize"},
    )
    builder.add_edge("save_output", END)

    return builder.compile()
