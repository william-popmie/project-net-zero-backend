from function_spec import FunctionSpec
from graph import build_graph


def _make_initial_state(spec: FunctionSpec, engine: str = "claude") -> dict:
    return {
        "spec": spec,
        "current_source": "",
        "baseline_emissions": 0.0,
        "current_emissions": 0.0,
        "test_passed": False,
        "attempt": 0,
        "max_attempts": 4,
        "last_test_output": "",
        "output_path": "",
        "success": False,
        "engine": engine,
        "inference_duration": 0.0,
        "inference_tokens": 0,
    }


def optimize_function(spec: FunctionSpec, engine: str = "claude") -> str:
    graph = build_graph(engine=engine)
    initial_state = _make_initial_state(spec, engine)
    final_state = graph.invoke(initial_state)
    return final_state["current_source"]


def run_engine(spec: FunctionSpec, engine: str) -> dict:
    """Run a single engine and return full result dict for benchmarking."""
    graph = build_graph(engine=engine)
    initial_state = _make_initial_state(spec, engine)
    final_state = graph.invoke(initial_state)
    return {
        "engine": engine,
        "function_name": spec.function_name,
        "optimized_source": final_state.get("current_source", ""),
        "baseline_emissions": final_state.get("baseline_emissions", 0.0),
        "current_emissions": final_state.get("current_emissions", 0.0),
        "test_passed": final_state.get("test_passed", False),
        "success": final_state.get("success", False),
        "attempts": final_state.get("attempt", 0),
        "inference_duration": final_state.get("inference_duration", 0.0),
        "inference_tokens": final_state.get("inference_tokens", 0),
    }


def compare_engines(spec: FunctionSpec) -> dict:
    """Run both Claude and Crusoe on the same function spec, return results."""
    claude_result = run_engine(spec, "claude")
    crusoe_result = run_engine(spec, "crusoe")
    return {
        "function_name": spec.function_name,
        "claude": claude_result,
        "crusoe": crusoe_result,
    }


if __name__ == "__main__":
    spec = FunctionSpec(
        function_name="is_prime",
        module_path="input-folder/mock_repo/math_utils.py",
        function_source="""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True""",
        test_source="""def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(4) == False
    assert is_prime(17) == True""",
    )
    result = optimize_function(spec)
    print(result)
