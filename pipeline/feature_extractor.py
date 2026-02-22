"""AST-based feature extractor for Python functions from scraped JSON files."""
import ast
import json
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class FunctionFeatures:
    function_id: str
    function_name: str
    source_file: str
    loc: int
    has_for_loop: bool
    has_while_loop: bool
    has_nested_loops: bool
    num_loops: int
    has_list_comp: bool
    has_dict_comp: bool
    has_generator: bool
    has_numpy: bool
    has_pandas: bool
    has_torch: bool
    has_tensorflow: bool
    has_sklearn: bool
    has_string_ops: bool
    num_function_calls: int
    num_arithmetic_ops: int
    cyclomatic_complexity: int
    category: str
    source_code: str


_STRING_OPS = {"join", "split", "strip", "replace", "format", "encode", "decode", "upper", "lower", "find"}
_ARITH_OP_TYPES = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv, ast.MatMult)
_DECISION_TYPES = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert, ast.IfExp, ast.comprehension)


def extract_module_imports(tree: ast.Module) -> set[str]:
    """Return top-level package names imported in the module."""
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def detect_category(imports: set[str]) -> str:
    """Map import set to a category string."""
    if imports & {"torch", "tensorflow", "keras", "sklearn", "lightgbm", "xgboost", "transformers"}:
        return "ml"
    if imports & {"pandas", "numpy", "scipy", "polars", "dask", "pyarrow"}:
        return "data_processing"
    if imports & {"flask", "fastapi", "django", "sqlalchemy", "requests", "aiohttp", "starlette"}:
        return "utility"
    return "other"


def _has_nested_loops(node: ast.AST) -> bool:
    """True if the subtree contains a loop nested inside another loop."""
    for child in ast.walk(node):
        if isinstance(child, (ast.For, ast.While)):
            for inner in ast.walk(child):
                if inner is not child and isinstance(inner, (ast.For, ast.While)):
                    return True
    return False


def _count_loops(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))


def _has_string_ops(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and n.func.attr in _STRING_OPS:
                return True
        if isinstance(n, ast.JoinedStr):
            return True
    return False


def _num_arith_ops(node: ast.AST) -> int:
    return sum(
        1 for n in ast.walk(node)
        if isinstance(n, ast.BinOp) and isinstance(n.op, _ARITH_OP_TYPES)
    )


def _num_calls(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Simplified McCabe metric: 1 + count of decision points."""
    return 1 + sum(1 for n in ast.walk(node) if isinstance(n, _DECISION_TYPES))


def _should_skip(func_node: ast.FunctionDef | ast.AsyncFunctionDef, loc: int, cc: int, source: str) -> bool:
    name = func_node.name
    if name.startswith("_") or name.startswith("test_"):
        return True
    if loc < 10 or loc > 150:
        return True
    if cc < 2:
        return True
    # Skip class methods (first arg is self or cls)
    args = func_node.args.args
    if args and args[0].arg in ("self", "cls"):
        return True
    # Skip functions that read from stdin — they block indefinitely in benchmarks
    if "input(" in source:
        return True
    return False


def extract_features_from_source(
    content: str,
    file_label: str,
    repo_label: str,
) -> list[FunctionFeatures]:
    """
    Parse a Python source string and extract FunctionFeatures for each eligible
    top-level function.

    file_label: path within the repo (e.g. "src/utils.py")
    repo_label: "owner/repo"
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports = extract_module_imports(tree)
    category = detect_category(imports)
    source_lines = content.splitlines()

    # Only collect top-level functions (direct children of the module)
    results: list[FunctionFeatures] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue

        loc = (node.end_lineno - node.lineno + 1) if node.end_lineno else 1
        cc = _cyclomatic_complexity(node)

        # Extract source code before skip check so we can inspect the body
        func_source_lines = source_lines[node.lineno - 1: node.end_lineno]
        func_source = textwrap.dedent("\n".join(func_source_lines))

        if _should_skip(node, loc, cc, func_source):
            continue

        num_loops = _count_loops(node)
        safe_path = file_label.replace("/", "__").replace("\\", "__")
        safe_repo = repo_label.replace("/", "__")
        func_id = f"{safe_repo}__{safe_path}__{node.name}"

        feat = FunctionFeatures(
            function_id=func_id,
            function_name=node.name,
            source_file=f"{repo_label}:{file_label}",
            loc=loc,
            has_for_loop=any(isinstance(n, ast.For) for n in ast.walk(node)),
            has_while_loop=any(isinstance(n, ast.While) for n in ast.walk(node)),
            has_nested_loops=_has_nested_loops(node),
            num_loops=num_loops,
            has_list_comp=any(isinstance(n, ast.ListComp) for n in ast.walk(node)),
            has_dict_comp=any(isinstance(n, ast.DictComp) for n in ast.walk(node)),
            has_generator=any(isinstance(n, ast.GeneratorExp) for n in ast.walk(node)),
            has_numpy=("numpy" in imports),
            has_pandas=("pandas" in imports),
            has_torch=("torch" in imports),
            has_tensorflow=("tensorflow" in imports),
            has_sklearn=("sklearn" in imports),
            has_string_ops=_has_string_ops(node),
            num_function_calls=_num_calls(node),
            num_arithmetic_ops=_num_arith_ops(node),
            cyclomatic_complexity=cc,
            category=category,
            source_code=func_source,
        )
        results.append(feat)

    return results


def run_extraction(raw_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Load all scraped JSON files, extract function features, save to CSV.
    Idempotent: if output_csv already exists, loads and returns it.
    """
    if output_csv.exists():
        print(f"[extractor] Loading existing {output_csv}")
        return pd.read_csv(output_csv)

    raw_files = sorted(raw_dir.glob("*.json"))
    print(f"[extractor] Processing {len(raw_files)} raw files...")

    all_features: list[FunctionFeatures] = []
    for raw_path in raw_files:
        try:
            record = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[extractor] Skipping {raw_path.name}: {e}")
            continue

        content = record.get("content", "")
        repo = record.get("repo", "unknown")
        path = record.get("path", raw_path.stem)
        features = extract_features_from_source(content, path, repo)
        all_features.extend(features)

    if not all_features:
        print("[extractor] Warning: no features extracted")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(f) for f in all_features])
    df = df.drop_duplicates(subset=["function_id"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[extractor] Saved {len(df)} functions to {output_csv}")
    return df
