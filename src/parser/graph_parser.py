from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env",
    "__pycache__", "build", "dist", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages",
    "node_modules", ".tox",
}

TEST_DIR_NAMES = {"tests", "test", "spec", "specs"}
TEST_FILE_PREFIXES = ("test_", "spec_")
TEST_FILE_SUFFIXES = ("_test.py", "_spec.py")


@dataclass
class FunctionInfo:
    id: str              # fully-qualified dotted path, e.g. "src.activations.ReLu.__call__"
    name: str            # bare function name, e.g. "__call__"
    qualified_name: str  # relative: "ReLu.__call__" or just "my_func"
    file_path: str       # relative POSIX path, e.g. "src/activations.py"
    line: int
    end_line: int
    source_code: str = ""


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _is_excluded(rel: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in rel.parts)


def _is_test_file(rel: Path) -> bool:
    lower_parts = {p.lower() for p in rel.parts}
    if lower_parts & TEST_DIR_NAMES:
        return True
    name = rel.name.lower()
    return name.startswith(TEST_FILE_PREFIXES) or name.endswith(TEST_FILE_SUFFIXES)


def discover_source_files(project_root: Path) -> list[Path]:
    result = []
    for path in sorted(project_root.rglob("*.py")):
        rel = path.relative_to(project_root)
        if _is_excluded(rel) or _is_test_file(rel):
            continue
        result.append(path)
    return result


def discover_test_files(project_root: Path) -> list[Path]:
    result = []
    for path in sorted(project_root.rglob("*.py")):
        rel = path.relative_to(project_root)
        if _is_excluded(rel):
            continue
        if _is_test_file(rel):
            result.append(path)
    return result


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def parse_file(file_path: Path) -> ast.AST | None:
    try:
        return ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError) as e:
        print(f"  [parser] skipping {file_path.name}: {e}")
        return None


def _file_to_module(rel: Path) -> str:
    """Convert a relative file path to a dotted module string, replacing hyphens."""
    parts = [p.replace("-", "_") for p in rel.with_suffix("").parts]
    return ".".join(parts)


def _collect_from_node(
    node: ast.AST,
    file_rel: Path,
    module_prefix: str,
    class_stack: list[str],
    out: list[FunctionInfo],
) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _collect_from_node(child, file_rel, module_prefix, [*class_stack, child.name], out)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified_name = ".".join([*class_stack, child.name])
            out.append(FunctionInfo(
                id=f"{module_prefix}.{qualified_name}",
                name=child.name,
                qualified_name=qualified_name,
                file_path=file_rel.as_posix(),
                line=child.lineno,
                end_line=child.end_lineno or child.lineno,
                source_code=ast.unparse(child),
            ))
            # recurse to pick up nested functions
            _collect_from_node(child, file_rel, module_prefix, [*class_stack, child.name], out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_functions(project_root: Path) -> list[FunctionInfo]:
    """Walk all source files under project_root and return every function/method."""
    root = project_root.resolve()

    source_files = discover_source_files(root)
    test_files = discover_test_files(root)

    print(f"  Source files ({len(source_files)}):")
    for f in source_files:
        print(f"    {f.relative_to(root)}")
    if test_files:
        print(f"  Test files ({len(test_files)}) — skipped for parsing:")
        for f in test_files:
            print(f"    {f.relative_to(root)}")

    functions: list[FunctionInfo] = []
    for file_path in source_files:
        tree = parse_file(file_path)
        if tree is None:
            continue
        rel = file_path.relative_to(root)
        module_prefix = _file_to_module(rel)
        _collect_from_node(tree, rel, module_prefix, [], functions)

    return functions


def find_requirements(project_root: Path) -> Path | None:
    """Return the requirements.txt path, searching project_root then its parent."""
    for candidate in [project_root / "requirements.txt",
                      project_root.parent / "requirements.txt"]:
        if candidate.exists():
            return candidate
    return None
