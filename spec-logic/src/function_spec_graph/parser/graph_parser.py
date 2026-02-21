from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
}

TEST_DIR_NAMES = {"tests", "test", "spec", "specs"}
TEST_FILE_PREFIXES = ("test_", "spec_")
TEST_FILE_SUFFIXES = ("_test.py", "_spec.py")


@dataclass(slots=True)
class FunctionNode:
    id: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    line: int


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: str


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, file_path: Path, module_path: str, kind: str) -> None:
        self.file_path = file_path
        self.module_path = module_path
        self.kind = kind
        self.scope_stack: list[str] = []
        self.collected: list[FunctionNode] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._collect_function(node)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._collect_function(node)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _collect_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        scope_prefix = ".".join(self.scope_stack)
        qualified_name = ".".join(
            part for part in (self.module_path, scope_prefix, node.name) if part
        )
        node_id = qualified_name

        self.collected.append(
            FunctionNode(
                id=node_id,
                kind=self.kind,
                name=node.name,
                qualified_name=qualified_name,
                file_path=self.file_path.as_posix(),
                line=node.lineno,
            )
        )


def discover_python_files(project_root: Path) -> list[Path]:
    discovered_files: list[Path] = []
    for path in project_root.rglob("*.py"):
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        discovered_files.append(path)
    return discovered_files


def is_test_file(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & TEST_DIR_NAMES:
        return True

    lower_name = path.name.lower()
    if lower_name.startswith(TEST_FILE_PREFIXES):
        return True
    return lower_name.endswith(TEST_FILE_SUFFIXES)


def path_to_module_path(project_root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(project_root)
    without_suffix = relative.with_suffix("")
    return ".".join(without_suffix.parts)


def parse_python_file(file_path: Path) -> ast.AST | None:
    try:
        source = file_path.read_text(encoding="utf-8")
        return ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None


def collect_called_function_names(tree_node: ast.AST) -> set[str]:
    called_names: set[str] = set()
    for node in ast.walk(tree_node):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)

    return called_names


def collect_function_call_map(module_tree: ast.AST, module_path: str) -> dict[str, set[str]]:
    call_map: dict[str, set[str]] = {}

    def walk(node: ast.AST, scope_stack: list[str]) -> None:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                walk(child, [*scope_stack, node.name])
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope_prefix = ".".join(scope_stack)
            qualified_name = ".".join(
                part for part in (module_path, scope_prefix, node.name) if part
            )
            call_map[qualified_name] = collect_called_function_names(node)

            for child in node.body:
                walk(child, [*scope_stack, node.name])
            return

        for child in ast.iter_child_nodes(node):
            walk(child, scope_stack)

    walk(module_tree, [])
    return call_map


def _calculate_coverage(
    source_nodes: list[FunctionNode],
    test_nodes: list[FunctionNode],
    edges: list[GraphEdge],
) -> dict[str, Any]:
    tested_ids: set[str] = {edge.source for edge in edges}
    untested_nodes = [node for node in source_nodes if node.id not in tested_ids]
    tested_nodes = [node for node in source_nodes if node.id in tested_ids]

    function_coverage = []
    for source_node in source_nodes:
        specs_for_function = [
            edge for edge in edges if edge.source == source_node.id
        ]
        function_coverage.append(
            {
                "function_id": source_node.id,
                "name": source_node.name,
                "qualified_name": source_node.qualified_name,
                "file_path": source_node.file_path,
                "line": source_node.line,
                "specs": [
                    {
                        "spec_id": edge.target,
                        "confidence": edge.confidence,
                        "relation": edge.relation,
                    }
                    for edge in specs_for_function
                ],
                "is_tested": len(specs_for_function) > 0,
            }
        )

    coverage_percentage = (
        (len(tested_nodes) / len(source_nodes) * 100)
        if source_nodes
        else 0
    )

    return {
        "total_functions": len(source_nodes),
        "tested_functions": len(tested_nodes),
        "untested_functions": len(untested_nodes),
        "coverage_percentage": round(coverage_percentage, 2),
        "function_coverage": function_coverage,
        "untested_list": [
            {
                "function_id": node.id,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "file_path": node.file_path,
                "line": node.line,
            }
            for node in untested_nodes
        ],
    }


def _build_edges(
    source_nodes: list[FunctionNode],
    test_nodes: list[FunctionNode],
    test_called_names: dict[str, set[str]],
) -> list[GraphEdge]:
    source_node_ids_by_name: dict[str, list[str]] = defaultdict(list)
    for source_node in source_nodes:
        source_node_ids_by_name[source_node.name].append(source_node.id)

    edges: list[GraphEdge] = []

    for test_node in test_nodes:
        matched_source_ids: set[str] = set()
        confidence = "direct_call"

        for called_name in test_called_names.get(test_node.id, set()):
            matched_source_ids.update(source_node_ids_by_name.get(called_name, []))

        if not matched_source_ids:
            confidence = "name_heuristic"
            lowered_test_name = test_node.name.lower()
            for prefix in ("test_", "should_", "it_", "spec_"):
                if lowered_test_name.startswith(prefix):
                    lowered_test_name = lowered_test_name[len(prefix) :]
                    break

            for source_name, source_ids in source_node_ids_by_name.items():
                if source_name.lower() in lowered_test_name:
                    matched_source_ids.update(source_ids)

        for source_id in sorted(matched_source_ids):
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=test_node.id,
                    relation="validated_by",
                    confidence=confidence,
                )
            )

    return edges


def build_graph(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()

    source_nodes: list[FunctionNode] = []
    test_nodes: list[FunctionNode] = []
    test_called_names: dict[str, set[str]] = {}

    for file_path in discover_python_files(root):
        module_tree = parse_python_file(file_path)
        if module_tree is None:
            continue

        module_path = path_to_module_path(root, file_path)
        is_test = is_test_file(file_path)

        collector = FunctionCollector(
            file_path=file_path.relative_to(root),
            module_path=module_path,
            kind="spec_function" if is_test else "project_function",
        )
        collector.visit(module_tree)

        if is_test:
            test_nodes.extend(collector.collected)
            call_map = collect_function_call_map(module_tree, module_path)
            for test_node in collector.collected:
                test_called_names[test_node.id] = call_map.get(test_node.id, set())
        else:
            source_nodes.extend(collector.collected)

    all_nodes = source_nodes + test_nodes
    edges = _build_edges(source_nodes, test_nodes, test_called_names)
    coverage = _calculate_coverage(source_nodes, test_nodes, edges)

    return {
        "metadata": {
            "project_root": root.as_posix(),
            "source_function_count": len(source_nodes),
            "spec_function_count": len(test_nodes),
            "edge_count": len(edges),
        },
        "nodes": [asdict(node) for node in all_nodes],
        "edges": [asdict(edge) for edge in edges],
        "coverage": coverage,
    }


def write_graph_json(graph: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def graph_to_html(graph: dict[str, Any]) -> str:
    coverage = graph["coverage"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    nodes_html = ""
    for node in nodes:
        node_type = "Project" if node["kind"] == "project_function" else "Test"
        is_tested = "✓" if node["kind"] == "spec_function" else (
            "✓" if any(edge["source"] == node["id"] for edge in edges) else "✗"
        )
        nodes_html += f"<tr><td>{node['qualified_name']}</td><td>{node['file_path']}:{node['line']}</td><td>{node_type}</td><td>{is_tested}</td></tr>"

    untested_html = ""
    for node in coverage["untested_list"]:
        untested_html += f"<li><code>{node['qualified_name']}</code> ({node['file_path']}:{node['line']})</li>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Function-Spec Coverage Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ padding: 15px; border-radius: 6px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; flex: 1; min-width: 150px; }}
        .stat strong {{ display: block; font-size: 24px; margin-bottom: 5px; }}
        .stat em {{ font-size: 12px; opacity: 0.9; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #667eea; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .untested {{ background-color: #ffe6e6; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 12px; }}
        .untested-list {{ list-style-type: none; padding: 0; }}
        .untested-list li {{ padding: 8px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; padding-left: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Function-Spec Coverage Report</h1>
        <div class="stats">
            <div class="stat">
                <strong>{coverage['coverage_percentage']}%</strong>
                <em>Coverage</em>
            </div>
            <div class="stat">
                <strong>{coverage['tested_functions']}/{coverage['total_functions']}</strong>
                <em>Functions Tested</em>
            </div>
            <div class="stat">
                <strong>{coverage['untested_functions']}</strong>
                <em>Functions Untested</em>
            </div>
        </div>
        
        <h2>All Functions</h2>
        <table>
            <tr><th>Function</th><th>Location</th><th>Type</th><th>Tested</th></tr>
            {nodes_html}
        </table>
        
        <h2>Untested Functions ({coverage['untested_functions']})</h2>
        <ul class="untested-list">
            {untested_html}
        </ul>
    </div>
</body>
</html>"""
    return html


def write_graph_html(graph: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph_to_html(graph), encoding="utf-8")


def graph_to_mermaid(graph: dict[str, Any]) -> str:
    def sanitize_mermaid_id(raw_value: str) -> str:
        allowed = []
        for character in raw_value:
            if character.isalnum() or character == "_":
                allowed.append(character)
            else:
                allowed.append("_")
        return "".join(allowed)

    lines = ["flowchart LR"]

    for node in graph["nodes"]:
        mermaid_id = sanitize_mermaid_id(node["id"])
        display_name = node["qualified_name"].replace('"', "'")
        if node["kind"] == "project_function":
            lines.append(f'    {mermaid_id}["{display_name}"]')
        else:
            lines.append(f'    {mermaid_id}(["{display_name}"])')

    for edge in graph["edges"]:
        source_mermaid_id = sanitize_mermaid_id(edge["source"])
        target_mermaid_id = sanitize_mermaid_id(edge["target"])
        lines.append(f"    {source_mermaid_id} -->|validated_by| {target_mermaid_id}")

    return "\n".join(lines) + "\n"


def write_graph_mermaid(graph: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph_to_mermaid(graph), encoding="utf-8")
