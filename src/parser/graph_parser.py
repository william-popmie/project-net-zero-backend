from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
class FunctionInfo:
    id: str
    name: str
    qualified_name: str
    file_path: str
    line: int
    end_line: int
    source_code: str = ""


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, file_path: Path, module_path: str, kind: str) -> None:
        self.file_path = file_path
        self.module_path = module_path
        self.kind = kind
        self.scope_stack: list[str] = []
        self.collected: list[FunctionInfo] = []

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
            FunctionInfo(
                id=node_id,
                name=node.name,
                qualified_name=qualified_name,
                file_path=self.file_path.as_posix(),
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )
        )


def discover_python_files(project_root: Path) -> list[Path]:
    discovered_files: list[Path] = []
    for path in project_root.rglob("*.py"):
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        discovered_files.append(path)
    return sorted(discovered_files)


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        logger.warning("Skipping %s: syntax error: %s", file_path, e)
        return None
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Skipping %s: %s", file_path, e)
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


def extract_function_source(file_path: Path, qualified_name: str) -> str:
    """Extract source code for a function by qualified name.

    Handles both top-level functions and class methods by trying
    increasingly short suffixes of the qualified name.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(file_path))

        parts = qualified_name.split(".")

        def navigate(node: ast.AST, remaining: list[str]) -> ast.AST | None:
            if not remaining:
                return None
            target = remaining[0]
            rest = remaining[1:]
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == target:
                        if not rest:
                            return child
                        result = navigate(child, rest)
                        if result is not None:
                            return result
            return None

        # Try from most specific (full path) to least (just function name),
        # so class methods are resolved before ambiguous bare names.
        for start in range(len(parts)):
            node = navigate(tree, parts[start:])
            if node is not None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.unparse(node)
    except Exception:
        pass

    return ""


def collect_functions(project_root: Path) -> list[FunctionInfo]:
    """Collect all non-test source functions from the project."""
    root = project_root.resolve()
    functions: list[FunctionInfo] = []

    for file_path in discover_python_files(root):
        # Use relative path so parent directories outside the project don't
        # accidentally trigger the test-file heuristic.
        if is_test_file(file_path.relative_to(root)):
            continue

        module_tree = parse_python_file(file_path)
        if module_tree is None:
            continue

        module_path = path_to_module_path(root, file_path)
        collector = FunctionCollector(
            file_path=file_path.relative_to(root),
            module_path=module_path,
            kind="project_function",
        )
        collector.visit(module_tree)
        functions.extend(collector.collected)

    return functions
