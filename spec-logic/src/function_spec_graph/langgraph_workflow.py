from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage


class WorkflowState(TypedDict):
    """State that flows through the LangGraph workflow."""
    
    # Input
    project_root: Path
    use_ai_matching: bool
    coverage_threshold: float
    max_spec_generation_iterations: int
    
    # Step 1: AST Parsing
    graph: dict[str, Any] | None
    ast_complete: bool
    
    # Step 2: Test Matching
    matching_complete: bool
    
    # Step 3: Test Execution & Coverage
    test_results: dict[str, Any] | None
    coverage_percentage: float
    coverage_passed: bool
    test_execution_complete: bool
    
    # Step 4: Spec Generation (if needed)
    spec_generation_iteration: int
    spec_generation_results: list[dict[str, Any]]
    
    # Final Output
    final_output_path: Path | None
    workflow_complete: bool
    
    # Error handling
    errors: list[str]


def parse_ast_node(state: WorkflowState) -> WorkflowState:
    """
    Node 1: Parse Python project into AST and build initial graph.
    """
    from .parser.graph_parser import build_graph
    
    print("\n" + "="*60)
    print("STEP 1: AST PARSING")
    print("="*60)
    
    try:
        graph = build_graph(
            state["project_root"],
            use_ai_matching=state["use_ai_matching"]
        )
        
        metadata = graph["metadata"]
        print(f"[OK] Parsed {metadata['source_function_count']} project functions")
        print(f"[OK] Found {metadata['spec_function_count']} spec functions")
        
        state["graph"] = graph
        state["ast_complete"] = True
        
    except Exception as e:
        print(f"[ERROR] AST parsing failed: {e}")
        state["errors"].append(f"AST parsing: {str(e)}")
        state["ast_complete"] = False
    
    return state


def match_tests_node(state: WorkflowState) -> WorkflowState:
    """
    Node 2: Match test functions to project functions (already done in build_graph).
    """
    print("\n" + "="*60)
    print("STEP 2: TEST MATCHING")
    print("="*60)
    
    if not state["graph"]:
        print("[ERROR] No graph available")
        state["errors"].append("Test matching: No graph available")
        return state
    
    coverage = state["graph"]["coverage"]
    edges = state["graph"]["edges"]
    
    print(f"[OK] Matched {len(edges)} test-function pairs")
    print(f"[OK] Coverage: {coverage['coverage_percentage']}% "
          f"({coverage['tested_functions']}/{coverage['total_functions']} tested)")
    
    state["matching_complete"] = True
    return state


def execute_tests_and_coverage_node(state: WorkflowState) -> WorkflowState:
    """
    Node 3: Run pytest with coverage to validate specs and measure code coverage.
    """
    print("\n" + "="*60)
    print("STEP 3: TEST EXECUTION & COVERAGE ANALYSIS")
    print("="*60)
    
    if not state["graph"]:
        print("[ERROR] No graph available")
        state["errors"].append("Test execution: No graph available")
        return state
    
    project_root = state["project_root"]
    
    try:
        # Run pytest with coverage
        print("[*] Running pytest with coverage.py...")
        
        result = subprocess.run(
            [
                "python", "-m", "pytest",
                "--cov=src",  # Cover the src directory
                "--cov-report=json",
                "--cov-report=term",
                "-v"
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check if tests passed
        tests_passed = result.returncode == 0
        
        print("\n--- PYTEST OUTPUT ---")
        print(result.stdout)
        if result.stderr:
            print("--- PYTEST STDERR ---")
            print(result.stderr)
        print("--- END OUTPUT ---\n")
        
        # Read coverage report
        coverage_file = project_root / "coverage.json"
        coverage_percentage = 0.0
        
        if coverage_file.exists():
            import json
            coverage_data = json.loads(coverage_file.read_text())
            coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
            print(f"[OK] Code coverage: {coverage_percentage:.1f}%")
        else:
            print("[!] No coverage.json found, using graph coverage instead")
            coverage_percentage = state["graph"]["coverage"]["coverage_percentage"]
        
        # Determine if coverage threshold is met
        threshold = state["coverage_threshold"]
        coverage_passed = coverage_percentage >= threshold and tests_passed
        
        state["test_results"] = {
            "tests_passed": tests_passed,
            "returncode": result.returncode,
            "output": result.stdout,
        }
        state["coverage_percentage"] = coverage_percentage
        state["coverage_passed"] = coverage_passed
        state["test_execution_complete"] = True
        
        if coverage_passed:
            print(f"[OK] Coverage threshold met: {coverage_percentage:.1f}% >= {threshold}%")
        else:
            if not tests_passed:
                print(f"[FAIL] Tests failed (exit code {result.returncode})")
            else:
                print(f"[FAIL] Coverage below threshold: {coverage_percentage:.1f}% < {threshold}%")
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Test execution timeout")
        state["errors"].append("Test execution: Timeout after 60s")
        state["test_execution_complete"] = False
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        state["errors"].append(f"Test execution: {str(e)}")
        state["test_execution_complete"] = False
    
    return state


def should_generate_specs(state: WorkflowState) -> str:
    """
    Conditional routing: Decide if we need to generate more specs.
    """
    if not state["test_execution_complete"]:
        return "output"  # Skip to output if tests couldn't run
    
    if state["coverage_passed"]:
        return "output"  # Go to final output
    
    # Check if we've hit max iterations
    if state["spec_generation_iteration"] >= state["max_spec_generation_iterations"]:
        print(f"\n[!] Max spec generation iterations reached ({state['max_spec_generation_iterations']})")
        return "output"
    
    return "generate_specs"  # Generate more specs


def generate_specs_node(state: WorkflowState) -> WorkflowState:
    """
    Node 4: Generate additional spec files using AI for untested/undercovered functions.
    """
    from .parser.ai_spec_generator import generate_specs_for_untested
    
    print("\n" + "="*60)
    print(f"STEP 4: AI SPEC GENERATION (Iteration {state['spec_generation_iteration'] + 1})")
    print("="*60)
    
    if not state["graph"]:
        print("[ERROR] No graph available")
        return state
    
    try:
        results = generate_specs_for_untested(
            state["graph"],
            state["project_root"],
            dry_run=False
        )
        
        state["spec_generation_results"].append(results)
        state["spec_generation_iteration"] += 1
        
        # Re-parse the project to include new specs
        print("\n[*] Re-parsing project with new specs...")
        from .parser.graph_parser import build_graph
        
        state["graph"] = build_graph(
            state["project_root"],
            use_ai_matching=state["use_ai_matching"]
        )
        
        # Re-run tests (loop back)
        # LangGraph will call execute_tests_and_coverage_node again
        state["test_execution_complete"] = False
        
    except Exception as e:
        print(f"[ERROR] Spec generation failed: {e}")
        state["errors"].append(f"Spec generation: {str(e)}")
    
    return state


def generate_output_node(state: WorkflowState) -> WorkflowState:
    """
    Node 5: Generate final output with AST tree structure.
    """
    from .parser.graph_parser import write_graph_json, write_graph_mermaid, write_graph_html
    
    print("\n" + "="*60)
    print("STEP 5: FINAL OUTPUT GENERATION")
    print("="*60)
    
    if not state["graph"]:
        print("[ERROR] No graph available for output")
        state["errors"].append("Output generation: No graph available")
        return state
    
    try:
        output_dir = Path("output/langgraph_workflow")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write outputs
        json_path = output_dir / "final_graph.json"
        mermaid_path = output_dir / "final_graph.mmd"
        html_path = output_dir / "final_coverage_report.html"
        
        write_graph_json(state["graph"], json_path)
        write_graph_mermaid(state["graph"], mermaid_path)
        write_graph_html(state["graph"], html_path)
        
        print(f"[OK] JSON output: {json_path}")
        print(f"[OK] Mermaid diagram: {mermaid_path}")
        print(f"[OK] HTML coverage report: {html_path}")
        
        # Generate AST summary
        ast_summary_path = output_dir / "ast_summary.txt"
        with open(ast_summary_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("AST TREE STRUCTURE SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for node in state["graph"]["nodes"]:
                if node["kind"] == "project_function":
                    f.write(f"Function: {node['qualified_name']}\n")
                    f.write(f"  File: {node['file_path']}:{node['line']}\n")
                    
                    # Find related specs
                    specs = [
                        edge for edge in state["graph"]["edges"]
                        if edge["source"] == node["id"]
                    ]
                    if specs:
                        f.write(f"  Specs ({len(specs)}):\n")
                        for edge in specs:
                            spec_node = next(
                                n for n in state["graph"]["nodes"]
                                if n["id"] == edge["target"]
                            )
                            f.write(f"    - {spec_node['qualified_name']} "
                                  f"(confidence: {edge['confidence']})\n")
                    else:
                        f.write("  Specs: [NONE - UNTESTED]\n")
                    f.write("\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("WORKFLOW SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Coverage: {state['coverage_percentage']:.1f}%\n")
            f.write(f"Tests passed: {state.get('test_results', {}).get('tests_passed', False)}\n")
            f.write(f"Spec generation iterations: {state['spec_generation_iteration']}\n")
            
            if state["errors"]:
                f.write(f"\nErrors encountered: {len(state['errors'])}\n")
                for error in state["errors"]:
                    f.write(f"  - {error}\n")
        
        print(f"[OK] AST summary: {ast_summary_path}")
        
        state["final_output_path"] = output_dir
        state["workflow_complete"] = True
        
    except Exception as e:
        print(f"[ERROR] Output generation failed: {e}")
        state["errors"].append(f"Output generation: {str(e)}")
    
    return state


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for the spec-logic pipeline.
    """
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("parse_ast", parse_ast_node)
    workflow.add_node("match_tests", match_tests_node)
    workflow.add_node("execute_tests", execute_tests_and_coverage_node)
    workflow.add_node("generate_specs", generate_specs_node)
    workflow.add_node("output", generate_output_node)
    
    # Define edges
    workflow.set_entry_point("parse_ast")
    workflow.add_edge("parse_ast", "match_tests")
    workflow.add_edge("match_tests", "execute_tests")
    
    # Conditional routing after test execution
    workflow.add_conditional_edges(
        "execute_tests",
        should_generate_specs,
        {
            "generate_specs": "generate_specs",
            "output": "output"
        }
    )
    
    # After spec generation, loop back to test execution
    workflow.add_edge("generate_specs", "execute_tests")
    
    # Output is the final node
    workflow.add_edge("output", END)
    
    return workflow.compile()


def run_workflow(
    project_root: Path,
    use_ai_matching: bool = False,
    coverage_threshold: float = 80.0,
    max_iterations: int = 3,
) -> WorkflowState:
    """
    Execute the complete LangGraph workflow.
    
    Args:
        project_root: Root directory of the Python project to analyze
        use_ai_matching: Whether to use Claude AI for test matching
        coverage_threshold: Minimum coverage percentage required (0-100)
        max_iterations: Maximum spec generation iterations
    
    Returns:
        Final workflow state with results
    """
    initial_state: WorkflowState = {
        "project_root": project_root,
        "use_ai_matching": use_ai_matching,
        "coverage_threshold": coverage_threshold,
        "max_spec_generation_iterations": max_iterations,
        "graph": None,
        "ast_complete": False,
        "matching_complete": False,
        "test_results": None,
        "coverage_percentage": 0.0,
        "coverage_passed": False,
        "test_execution_complete": False,
        "spec_generation_iteration": 0,
        "spec_generation_results": [],
        "final_output_path": None,
        "workflow_complete": False,
        "errors": [],
    }
    
    print("\n" + "="*60)
    print("LANGGRAPH WORKFLOW STARTED")
    print("="*60)
    print(f"Project: {project_root}")
    print(f"AI Matching: {use_ai_matching}")
    print(f"Coverage Threshold: {coverage_threshold}%")
    print(f"Max Iterations: {max_iterations}")
    
    workflow = build_workflow()
    
    final_state = workflow.invoke(initial_state)
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED")
    print("="*60)
    print(f"Final Coverage: {final_state['coverage_percentage']:.1f}%")
    print(f"Output Directory: {final_state['final_output_path']}")
    
    if final_state["errors"]:
        print(f"\nErrors encountered: {len(final_state['errors'])}")
        for error in final_state["errors"]:
            print(f"  - {error}")
    
    return final_state
