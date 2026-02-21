from __future__ import annotations

import argparse
from pathlib import Path

from .langgraph_workflow import run_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="function-spec-workflow",
        description=(
            "Run the complete LangGraph workflow: AST parsing → Test matching → "
            "Test execution → Coverage validation → Spec generation (if needed) → Output."
        ),
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the Python project that should be analyzed.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum code coverage percentage required (default: 80.0).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum spec generation iterations (default: 3).",
    )
    parser.add_argument(
        "--use-ai-matching",
        action="store_true",
        help="Use Claude AI to enhance test-function matching (requires ANTHROPIC_API_KEY).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_path = args.project_path.resolve()
    if not project_path.exists() or not project_path.is_dir():
        parser.error(f"Project path does not exist or is not a folder: {project_path}")

    final_state = run_workflow(
        project_root=project_path,
        use_ai_matching=args.use_ai_matching,
        coverage_threshold=args.coverage_threshold,
        max_iterations=args.max_iterations,
    )
    
    # Exit with error code if workflow failed
    if final_state["errors"] and not final_state["workflow_complete"]:
        exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()
