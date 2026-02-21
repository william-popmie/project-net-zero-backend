from function_spec import FunctionSpec
from graph import build_graph


def optimize_function(spec: FunctionSpec) -> str:
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
        "output_path": "",
        "success": False,
    }
    final_state = graph.invoke(initial_state)
    return final_state["current_source"]
