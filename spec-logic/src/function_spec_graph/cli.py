from __future__ import annotations

import argparse
from pathlib import Path

from .parser.graph_parser import build_graph, write_graph_json, write_graph_mermaid, write_graph_html
from .parser.ai_spec_generator import generate_specs_for_untested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="function-spec-graph",
        description=(
            "Generate a graph with project functions and spec/test functions that validate them."
        ),
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the Python project that should be analyzed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/function_spec_graph.json"),
        help="Output path for JSON graph data (includes coverage).",
    )
    parser.add_argument(
        "--output-mermaid",
        type=Path,
        default=Path("output/function_spec_graph.mmd"),
        help="Output path for Mermaid diagram markup.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("output/coverage_report.html"),
        help="Output path for HTML coverage report.",
    )
    parser.add_argument(
        "--use-ai-matching",
        action="store_true",
        help="Use Claude AI to enhance test-function matching (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--generate-missing-specs",
        action="store_true",
        help="Auto-generate pytest tests for untested functions using Claude AI (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When used with --generate-missing-specs, shows what would be generated without writing files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_path = args.project_path.resolve()
    if not project_path.exists() or not project_path.is_dir():
        parser.error(f"Project path does not exist or is not a folder: {project_path}")

    graph = build_graph(project_path, use_ai_matching=args.use_ai_matching)
    write_graph_json(graph, args.output_json)
    write_graph_mermaid(graph, args.output_mermaid)
    write_graph_html(graph, args.output_html)

    metadata = graph["metadata"]
    coverage = graph["coverage"]
    print(
        "Graph generated: "
        f"{metadata['source_function_count']} project functions, "
        f"{metadata['spec_function_count']} spec functions, "
        f"{metadata['edge_count']} links"
    )
    print(f"Coverage: {coverage['coverage_percentage']}% ({coverage['tested_functions']}/{coverage['total_functions']} tested)")
    print(f"JSON: {args.output_json}")
    print(f"Mermaid: {args.output_mermaid}")
    print(f"HTML Report: {args.output_html}")
    
    # Generate missing specs if requested
    if args.generate_missing_specs:
        print("\n" + "="*60)
        print("AI SPEC GENERATION")
        print("="*60 + "\n")
        
        results = generate_specs_for_untested(
            graph,
            project_path,
            dry_run=args.dry_run
        )
        
        print("\n" + "-"*60)
        print(f"[OK] Generated {results['generated_count']} test functions")
        if results['failed_count'] > 0:
            print(f"[ERROR] Failed {results['failed_count']} generations")
        if results['generated_files']:
            print(f"\nGenerated files:")
            for file in results['generated_files']:
                print(f"  - {file}")
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        print("-"*60)



if __name__ == "__main__":
    main()
