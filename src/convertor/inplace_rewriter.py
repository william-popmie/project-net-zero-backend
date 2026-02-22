"""Rewrite optimized functions directly into their original source files."""

from __future__ import annotations

import ast
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


def rewrite_functions_inplace(
    project_root: Path | str,
    optimizer_results: list[dict[str, Any]],
) -> list[str]:
    """
    Splice optimized function code back into the original source files.

    Args:
        project_root: Root directory of the project copy.
        optimizer_results: List of optimizer result dicts, each containing
            at least ``file``, ``name``, ``success``, and ``optimized_source``.

    Returns:
        List of file paths that were modified.
    """
    project_root = Path(project_root).resolve()

    # Group successful optimizations by source file.
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in optimizer_results:
        if not entry.get("success"):
            continue
        source = entry.get("optimized_source", "").strip()
        if not source:
            continue
        by_file[entry["file"]].append(entry)

    modified: list[str] = []

    for source_file, funcs in by_file.items():
        file_path = project_root / source_file
        if not file_path.exists():
            print(f"[rewriter] skipping missing file: {file_path}")
            continue

        original_text = file_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)

        tree = ast.parse(original_text, filename=str(file_path))

        # Collect (start_line, end_line, new_code) replacements.
        replacements: list[tuple[int, int, str]] = []
        for func_entry in funcs:
            node = _find_function_node(tree, func_entry["name"])
            if node is None:
                print(
                    f"[rewriter] could not locate {func_entry['name']!r} "
                    f"in {file_path}"
                )
                continue

            # AST lines are 1-indexed; we work with 0-indexed.
            start = node.lineno - 1
            end = node.end_lineno  # exclusive (end_lineno is 1-indexed inclusive)

            original_indent = len(lines[start]) - len(lines[start].lstrip())
            new_code = _reindent(func_entry["optimized_source"], original_indent)

            replacements.append((start, end, new_code))

        if not replacements:
            continue

        # Sort bottom-to-top so earlier line numbers remain valid.
        replacements.sort(key=lambda r: r[0], reverse=True)

        for start, end, new_code in replacements:
            new_lines = [ln + "\n" for ln in new_code.splitlines()]
            lines[start:end] = new_lines

        file_path.write_text("".join(lines), encoding="utf-8")
        modified.append(str(file_path))
        print(f"[rewriter] updated {file_path}")

    return modified


def _find_function_node(
    tree: ast.Module,
    qualified_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """
    Locate a function/method node by its (possibly qualified) name.

    Supports simple names like ``"my_func"`` and class-qualified names like
    ``"MyClass.my_method"``.
    """
    parts = qualified_name.split(".")

    if len(parts) == 1:
        # Top-level function.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == parts[0]:
                    return node
    elif len(parts) == 2:
        # Class method.
        class_name, method_name = parts
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name == method_name:
                            return child

    return None


def _reindent(source: str, target_indent: int) -> str:
    """
    Re-indent *source* so that its first line sits at *target_indent* spaces.

    The relative indentation of subsequent lines is preserved.
    """
    dedented = textwrap.dedent(source)
    if target_indent == 0:
        return dedented
    return textwrap.indent(dedented, " " * target_indent)
