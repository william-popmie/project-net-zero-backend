from .function_spec import FunctionSpec
from .graph import build_graph


def optimize_function(spec: FunctionSpec) -> dict:
    graph = build_graph()
    initial_state = {
        "spec": spec,
        "current_source": "",
        "baseline_emissions": 0.0,
        "current_emissions": 0.0,
        "test_passed": False,
        "attempt": 0,
        "max_attempts": 4,
        "last_test_output": "",
        "success": False,
    }
    final_state = graph.invoke(initial_state)
    return {
        "optimized_source": final_state["current_source"],
        "baseline_emissions": final_state["baseline_emissions"],
        "optimized_emissions": final_state["current_emissions"],
        "success": final_state["success"],
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
