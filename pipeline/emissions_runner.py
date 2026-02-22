"""Emissions measurement: baseline scan (Phase 1) + top-20 optimizer (Phase 2)."""
import ast
import concurrent.futures
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

# sys.path setup — must precede any src/ imports
_REPO_ROOT = Path(__file__).parent.parent
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

from optimizer_logic.emissions import measure_emissions_for_source  # noqa: E402
from optimizer_logic.function_spec import FunctionSpec  # noqa: E402
from optimizer_logic.optimizer import optimize_function  # noqa: E402

# ── Subprocess timeout patch ──────────────────────────────────────────────────
# measure_emissions_for_source calls subprocess.run() with no timeout, so it
# can block indefinitely. We patch subprocess.run here to enforce a hard limit.
# When it fires, subprocess.TimeoutExpired propagates up as RuntimeError, which
# our except-clauses catch and treat as a failed measurement.
import subprocess as _sp

_original_subprocess_run = _sp.run


def _timed_subprocess_run(*args, **kwargs):
    if "timeout" not in kwargs:
        kwargs["timeout"] = 28  # slightly under our outer 30s limit
    return _original_subprocess_run(*args, **kwargs)


_sp.run = _timed_subprocess_run

# Schema for emissions.csv
_BOOL_FEATURE_COLS = [
    "has_for_loop", "has_while_loop", "has_nested_loops",
    "has_list_comp", "has_dict_comp", "has_generator",
    "has_numpy", "has_pandas", "has_torch", "has_tensorflow",
    "has_sklearn", "has_string_ops",
]

EMISSIONS_COLS = [
    "function_id", "function_name", "source_file", "category", "loc",
    *_BOOL_FEATURE_COLS,
    "num_loops", "num_function_calls", "num_arithmetic_ops", "cyclomatic_complexity",
    "baseline_emissions_kg", "optimized_emissions_kg", "reduction_pct",
    "optimizer_ran", "error", "source_code", "optimized_source",
]


# ── Smart benchmark call generation ──────────────────────────────────────────

def _value_for_annotation(ann: ast.expr | None) -> tuple[str, str]:
    """
    Return (value_literal, import_line) for a type annotation node.
    import_line is "" if no extra import is needed.
    """
    if ann is None:
        return "97", ""

    # Unwrap the annotation to a type name string
    if isinstance(ann, ast.Name):
        name = ann.id
    elif isinstance(ann, ast.Attribute):
        name = ann.attr
    elif isinstance(ann, ast.Subscript):
        # e.g. List[int], Optional[str], Union[int, str]
        inner = ann.value
        if isinstance(inner, ast.Name):
            name = inner.id
        elif isinstance(inner, ast.Attribute):
            name = inner.attr
        else:
            name = ""
        if name == "Optional":
            return "None", ""
    elif isinstance(ann, ast.Constant) and ann.value is None:
        return "None", ""
    else:
        name = ""

    n = name.lower()

    # Numeric
    if n in ("int", "integer", ""):
        return "97", ""
    if n in ("float", "double", "number", "real"):
        return "1.0", ""
    if n in ("complex",):
        return "complex(1, 2)", ""

    # String / bytes
    if n in ("str", "string"):
        return '"hello"', ""
    if n in ("bytes", "bytearray"):
        return 'b"hello"', ""

    # Boolean
    if n in ("bool", "boolean"):
        return "True", ""

    # Collections
    if n in ("list", "sequence", "mutablesequence", "iterable", "iterator", "collection"):
        return "[1, 2, 3]", ""
    if n in ("dict", "mapping", "mutablemapping", "defaultdict", "ordereddict"):
        return '{"a": 1}', ""
    if n in ("set", "frozenset", "mutableset"):
        return "{1, 2, 3}", ""
    if n in ("tuple",):
        return "(1, 2, 3)", ""
    if n in ("deque",):
        return "collections.deque([1, 2, 3])", "import collections"

    # Optional / Any
    if n in ("optional", "union", "any"):
        return "None", ""
    if n in ("none", "nonetype"):
        return "None", ""

    # Numpy
    if n in ("ndarray", "array", "matrix"):
        return "np.array([1.0, 2.0, 3.0])", "import numpy as np"
    if n in ("int8", "int16", "int32", "int64", "float16", "float32", "float64"):
        return f"np.{n}(1)", "import numpy as np"

    # Pandas
    if n == "dataframe":
        return 'pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})', "import pandas as pd"
    if n == "series":
        return "pd.Series([1.0, 2.0, 3.0])", "import pandas as pd"
    if n == "index":
        return "pd.Index([1, 2, 3])", "import pandas as pd"

    # Path / IO
    if n in ("path", "purepath"):
        return 'Path(".")', "from pathlib import Path"
    if n in ("textio", "io", "bufferedreader", "bufferedwriter"):
        return "None", ""

    # Callables / functions
    if n in ("callable", "function"):
        return "(lambda x: x)", ""

    # Torch / TF / complex ML objects → skip with None
    if n in ("tensor", "module", "parameter", "device"):
        return "None", ""

    # Unknown → fall back to 97
    return "97", ""


def _value_for_param_name(param: str) -> str | None:
    """
    Return a default value literal based on the parameter *name* alone.
    Called as a fallback when the annotation gives no useful type info.
    """
    n = param.lower()
    # Integer counters / sizes
    if n in ("n", "k", "i", "j", "m", "count", "num", "size", "length",
             "max", "min", "limit", "idx", "index", "level", "depth", "width",
             "height", "steps", "iters", "iterations", "epoch", "batch_size"):
        return "97"
    # Boolean flags
    if any(n.startswith(p) for p in ("is_", "has_", "use_", "enable", "disable",
                                      "verbose", "debug", "force", "dry_run", "flag")):
        return "True"
    # File / path parameters — use current directory (safe, always exists)
    if any(kw in n for kw in ("path", "file", "dir", "folder", "filename",
                               "filepath", "directory", "output_dir", "input_dir")):
        return '"."'
    # URL / endpoint
    if any(kw in n for kw in ("url", "uri", "endpoint", "host", "base_url")):
        return '"http://localhost"'
    # Generic string names (name, label, key, tag, text, …)
    if any(n.endswith(sfx) for sfx in ("_name", "_str", "_text", "_label",
                                        "_key", "_tag", "_prefix", "_suffix",
                                        "_title", "_content", "_id")):
        return '"hello"'
    if n in ("text", "string", "content", "message", "query", "line",
             "token", "word", "char", "sep", "delimiter"):
        return '"hello"'
    # Numeric floats
    if n in ("rate", "ratio", "alpha", "beta", "gamma", "threshold",
             "epsilon", "lr", "learning_rate", "weight", "score"):
        return "1.0"
    return None


def _generate_smart_call(source: str, func_name: str) -> tuple[str, str]:
    """
    Inspect function signature to build a benchmark call that is more likely
    to succeed than the default all-97 approach.

    Returns (extra_imports, benchmark_call).
    extra_imports: newline-joined import statements to prepend to function_source.
    benchmark_call: the call expression string.
    """
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != func_name:
                continue

            args = node.args
            positional = args.args          # regular args
            defaults = args.defaults        # defaults for the LAST N positional args
            required_count = len(positional) - len(defaults)

            # If every arg has a default, call with no args
            if required_count == 0:
                return "", f"{func_name}()"

            imports_needed: list[str] = []
            call_parts: list[str] = []

            for i in range(required_count):
                arg = positional[i]
                val, imp = _value_for_annotation(arg.annotation)
                # If annotation gives no useful hint (falls back to "97"),
                # try the parameter name for a better default.
                if val == "97" and arg.annotation is None:
                    name_val = _value_for_param_name(arg.arg)
                    if name_val is not None:
                        val = name_val
                call_parts.append(val)
                if imp and imp not in imports_needed:
                    imports_needed.append(imp)

            extra_imports = "\n".join(imports_needed)
            benchmark_call = f"{func_name}({', '.join(call_parts)})"
            return extra_imports, benchmark_call

    except Exception:
        pass

    return "", f"{func_name}(97)"


# ── Measurement helpers ───────────────────────────────────────────────────────

# Common stdlib names used in function bodies but not re-imported in the
# standalone benchmark script (the file's module-level imports are stripped).
_STDLIB_PREAMBLE = """\
import abc
import argparse
import ast
import bisect
import collections
import collections.abc
import contextlib
import copy
import csv
import dataclasses
import datetime
import enum
import functools
import gc
import glob
import hashlib
import heapq
import inspect
import io
import itertools
import json
import logging
import math
import operator
import os
import pathlib
import queue
import random
import re
import shutil
import signal
import string
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import typing
import unicodedata
import uuid
import warnings
from collections import defaultdict, OrderedDict, Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple, Type, Union
try:
    import psutil
except ImportError:
    pass
"""


def _detect_library_imports(source: str) -> str:
    """
    Scan function source for known library aliases and return needed imports.
    Wrapped in try/except so a missing optional library doesn't crash the script.
    """
    lines: list[str] = []

    if "np." in source or "numpy" in source.lower():
        lines.append("try:\n    import numpy as np\nexcept ImportError:\n    pass")
    if "pd." in source or "pandas" in source.lower():
        lines.append("try:\n    import pandas as pd\nexcept ImportError:\n    pass")
    if "torch" in source:
        lines.append("try:\n    import torch\nexcept ImportError:\n    pass")
    if "sklearn" in source:
        lines.append("try:\n    import sklearn\nexcept ImportError:\n    pass")
    if "scipy" in source:
        lines.append("try:\n    import scipy\nexcept ImportError:\n    pass")
    if "PIL" in source or "Image" in source:
        lines.append("try:\n    from PIL import Image\nexcept ImportError:\n    pass")
    if "psutil" in source:
        lines.append("try:\n    import psutil\nexcept ImportError:\n    pass")
    if "git" in source and "Repo" in source:
        lines.append("try:\n    from git import Repo\nexcept ImportError:\n    pass")

    return "\n".join(lines)


def _build_augmented_source(source: str, extra_call_imports: str = "") -> str:
    """Prepend stdlib preamble + detected library imports to a function source string."""
    library_imports = _detect_library_imports(source)
    parts = [_STDLIB_PREAMBLE]
    if library_imports:
        parts.append(library_imports)
    if extra_call_imports:
        parts.append(extra_call_imports)
    return "\n".join(parts) + "\n\n" + source


def _prepare_full_module(full_content: str) -> str:
    """
    Prepare a full module file for use as a standalone benchmark source.
    Uses AST to accurately identify import statement boundaries (including
    multi-line imports), then wraps each in try/except.
    Also removes relative imports and __main__ blocks.
    """
    try:
        tree = ast.parse(full_content)
    except SyntaxError:
        return full_content

    lines = full_content.splitlines()
    # Collect line ranges of top-level imports and __main__ if blocks to rewrite
    rewrites: list[tuple[int, int, str]] = []  # (start_line, end_line, replacement)

    for node in ast.iter_child_nodes(tree):
        start = node.lineno - 1  # 0-indexed
        end = (node.end_lineno or node.lineno)  # exclusive

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Skip relative imports
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                rewrites.append((start, end, "pass  # relative import removed"))
                continue
            # Wrap in try/except
            import_src = "\n".join(lines[start:end])
            indented = "\n    ".join(import_src.splitlines())
            rewrites.append((start, end, f"try:\n    {indented}\nexcept Exception:\n    pass"))

        elif isinstance(node, ast.If):
            # Remove if __name__ == '__main__' blocks
            test = node.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)):
                rewrites.append((start, end, "# __main__ block removed"))

    if not rewrites:
        return full_content

    # Apply rewrites from bottom to top to preserve line numbers
    rewrites.sort(key=lambda r: r[0], reverse=True)
    result = lines[:]
    for start, end, replacement in rewrites:
        result[start:end] = replacement.splitlines()

    return "\n".join(result)


def _get_full_module_content(source_file: str, raw_dir: Path) -> str | None:
    """
    Load the full source file content from the raw JSON archive.
    source_file format: "owner/repo:path/to/file.py"
    """
    try:
        if ":" not in source_file:
            return None
        repo_part, path_part = source_file.split(":", 1)
        owner, repo = repo_part.split("/", 1)
        safe_path = path_part.replace("/", "__").replace("\\", "__")
        json_path = raw_dir / f"{owner}__{repo}__{safe_path}.json"
        if not json_path.exists():
            return None
        record = json.loads(json_path.read_text(encoding="utf-8"))
        return record.get("content")
    except Exception:
        return None


def _try_call(augmented_source: str, name: str, benchmark_call: str, timeout: int) -> float | None:
    """Run one measurement attempt in a thread with a timeout.
    Does NOT block waiting for the thread after a timeout — returns immediately."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(measure_emissions_for_source, augmented_source, name, benchmark_call)
    try:
        result = future.result(timeout=timeout)
        ex.shutdown(wait=False)
        return result
    except Exception:
        # TimeoutError or RuntimeError — abandon the thread, don't block
        ex.shutdown(wait=False)
        return None


def _fallback_string_call(source: str, name: str) -> str:
    """
    Generate a benchmark call using "true" for all unannotated required args.
    "true" works for bool-string functions; falls back to "hello" in the loop.
    """
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != name:
                continue
            args = node.args
            required_count = len(args.args) - len(args.defaults)
            if required_count == 0:
                return f"{name}()"
            parts = []
            for arg in args.args[:required_count]:
                if arg.annotation is None:
                    parts.append('"true"')
                else:
                    val, _ = _value_for_annotation(arg.annotation)
                    parts.append(val)
            return f"{name}({', '.join(parts)})"
    except Exception:
        pass
    return f'{name}("true")'


def _measure_with_timeout(
    source: str,
    name: str,
    timeout: int = 30,
    source_file: str = "",
    raw_dir: Path | None = None,
) -> float | None:
    """
    Try up to three benchmark strategies:
      1. Preamble + function source, annotation-based call
      2. Preamble + function source, string-fallback call
      3. Full module file as context (all module-level names available)
    Returns None only if all strategies fail or time out.
    """
    # Skip functions that call input() — they block stdin and will always hang.
    if "input(" in source:
        return None

    extra_imports, smart_call = _generate_smart_call(source, name)
    augmented = _build_augmented_source(source, extra_imports)
    fallback_call = _fallback_string_call(source, name)

    # Attempt 1: numeric/annotation call
    result = _try_call(augmented, name, smart_call, timeout)
    if result is not None:
        return result

    # Attempt 2: string fallback
    if fallback_call != smart_call:
        result = _try_call(augmented, name, fallback_call, timeout)
        if result is not None:
            return result

    # Attempt 3: full module context (resolves module-level name errors)
    if raw_dir and source_file:
        full_content = _get_full_module_content(source_file, raw_dir)
        if full_content:
            # Prepend stdlib preamble so module-shadowed names (e.g. logging) still resolve
            module_source = _STDLIB_PREAMBLE + "\n\n" + _prepare_full_module(full_content)
            # Try smart call then string fallback — deduplicated to avoid wasted attempts
            for call in dict.fromkeys([smart_call, fallback_call]):
                result = _try_call(module_source, name, call, timeout)
                if result is not None:
                    return result

    return None


def _load_successful_ids(emissions_csv: Path) -> set[str]:
    """
    Return function_ids that already have a successful baseline measurement.
    Failed rows (empty baseline_emissions_kg) are NOT included so they will
    be retried on the next run.
    """
    if not emissions_csv.exists():
        return set()
    try:
        df = pd.read_csv(emissions_csv, usecols=["function_id", "baseline_emissions_kg"])
        df["baseline_emissions_kg"] = pd.to_numeric(df["baseline_emissions_kg"], errors="coerce")
        successful = df[df["baseline_emissions_kg"].notna()]
        return set(successful["function_id"].astype(str).tolist())
    except Exception:
        return set()


def _drop_failed_rows(emissions_csv: Path) -> None:
    """
    Rewrite emissions_csv keeping only rows with a valid baseline measurement.
    This prevents duplicate rows when retrying failed functions.
    """
    if not emissions_csv.exists():
        return
    try:
        df = pd.read_csv(emissions_csv)
        df["baseline_emissions_kg"] = pd.to_numeric(df["baseline_emissions_kg"], errors="coerce")
        df_clean = df[df["baseline_emissions_kg"].notna()]
        df_clean.to_csv(emissions_csv, index=False)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def run_baseline_scan(
    functions_csv: Path,
    emissions_csv: Path,
    timeout_secs: int = 30,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Phase 1: Measure baseline emissions for all functions in functions_csv.
    Writes results incrementally to emissions_csv (one row per function).
    Retries previously failed measurements; skips only successful ones.
    Returns the complete emissions DataFrame.
    """
    if not functions_csv.exists():
        raise FileNotFoundError(f"functions.csv not found: {functions_csv}")

    # Drop any previously-failed rows so we can retry them cleanly
    _drop_failed_rows(emissions_csv)

    df = pd.read_csv(functions_csv)
    already_done = _load_successful_ids(emissions_csv)
    pending = df[~df["function_id"].astype(str).isin(already_done)]

    print(
        f"[runner] Baseline scan: {len(pending)} functions to measure "
        f"({len(already_done)} already succeeded, {len(df)} total)"
    )

    write_header = not emissions_csv.exists()
    emissions_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(emissions_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EMISSIONS_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, (_, row) in enumerate(pending.iterrows()):
            func_id = str(row["function_id"])
            func_name = str(row["function_name"])
            source = str(row.get("source_code", ""))

            print(f"[runner] ({i + 1}/{len(pending)}) {func_name}", flush=True)

            error_msg = None
            emissions = None
            try:
                emissions = _measure_with_timeout(
                    source, func_name,
                    timeout=timeout_secs,
                    source_file=str(row.get("source_file", "")),
                    raw_dir=raw_dir,
                )
                if emissions is None:
                    error_msg = "timeout_or_error"
            except Exception as e:
                error_msg = str(e)[:200]

            record: dict = {}
            for col in EMISSIONS_COLS:
                val = row.get(col, "")
                record[col] = "" if (isinstance(val, float) and str(val) == "nan") else val

            record["baseline_emissions_kg"] = emissions if emissions is not None else ""
            record["optimized_emissions_kg"] = ""
            record["reduction_pct"] = ""
            record["optimizer_ran"] = False
            record["error"] = error_msg or ""
            record["optimized_source"] = ""

            if emissions is not None:
                print(f"          -> {emissions:.3e} kg CO2eq", flush=True)
            else:
                print(f"          -> failed ({error_msg})", flush=True)

            writer.writerow(record)
            f.flush()

    return pd.read_csv(emissions_csv)


def run_top20_optimizer(
    emissions_csv: Path,
    functions_csv: Path,
    use_hints: bool = False,
) -> pd.DataFrame:
    """
    Phase 2: Run the full Claude optimizer on the top-20 most expensive functions.
    Stratified selection: top-10 from 'ml' + top-10 from 'data_processing'.
    Updates emissions_csv rows with optimized_emissions_kg, reduction_pct, optimized_source.
    """
    df = pd.read_csv(emissions_csv)
    funcs_df = pd.read_csv(functions_csv)

    df["baseline_emissions_kg"] = pd.to_numeric(df["baseline_emissions_kg"], errors="coerce")
    df["loc"] = pd.to_numeric(df["loc"], errors="coerce")
    df["cyclomatic_complexity"] = pd.to_numeric(df["cyclomatic_complexity"], errors="coerce")

    valid = df[
        df["baseline_emissions_kg"].notna()
        & (df["cyclomatic_complexity"] >= 2)
        & (df["loc"] >= 10)
        & (df["optimizer_ran"].astype(str) != "True")
    ].copy()

    ml_top = valid[valid["category"] == "ml"].nlargest(10, "baseline_emissions_kg")
    dp_top = valid[valid["category"] == "data_processing"].nlargest(10, "baseline_emissions_kg")
    selected_ids = set(ml_top["function_id"]).union(dp_top["function_id"])

    remaining_slots = 20 - len(selected_ids)
    if remaining_slots > 0:
        others = valid[~valid["function_id"].isin(selected_ids)].nlargest(
            remaining_slots, "baseline_emissions_kg"
        )
        top20 = pd.concat([ml_top, dp_top, others]).head(20)
    else:
        top20 = pd.concat([ml_top, dp_top]).head(20)

    print(f"[runner] Top-20 optimizer: {len(top20)} functions selected")

    source_lookup = dict(zip(funcs_df["function_id"].astype(str), funcs_df["source_code"].astype(str)))

    updates: dict[str, dict] = {}
    for _, row in top20.iterrows():
        func_id = str(row["function_id"])
        func_name = str(row["function_name"])
        source = source_lookup.get(func_id, str(row.get("source_code", "")))

        if not source or not source.strip():
            print(f"[runner] Skipping {func_name}: no source code")
            continue

        stub_test = f"def test_{func_name}_runs():\n    pass\n"

        spec = FunctionSpec(
            function_name=func_name,
            module_path=str(row.get("source_file", "pipeline/data/raw/unknown.py")),
            function_source=source,
            test_source=stub_test,
        )

        baseline = float(row["baseline_emissions_kg"])
        print(f"[runner] Optimizing: {func_name} (baseline: {baseline:.2e} kg CO2eq)")

        try:
            result = optimize_function(spec, use_hints=use_hints)
            optimized_emissions = result.get("optimized_emissions", 0.0)

            if baseline > 0 and optimized_emissions is not None:
                reduction = (baseline - optimized_emissions) / baseline * 100
            else:
                reduction = 0.0

            updates[func_id] = {
                "optimized_emissions_kg": optimized_emissions,
                "reduction_pct": reduction,
                "optimizer_ran": True,
                "optimized_source": result.get("optimized_source", source),
                "error": "",
            }
            print(f"[runner] {func_name}: {reduction:.1f}% reduction")

        except Exception as e:
            print(f"[runner] {func_name} optimizer failed: {e}")
            updates[func_id] = {
                "optimized_emissions_kg": float("nan"),
                "reduction_pct": float("nan"),
                "optimizer_ran": True,
                "optimized_source": "",
                "error": str(e)[:200],
            }

    if updates:
        df_out = pd.read_csv(emissions_csv)
        for func_id, upd in updates.items():
            mask = df_out["function_id"].astype(str) == func_id
            for col, val in upd.items():
                if col in df_out.columns:
                    df_out.loc[mask, col] = val
        df_out.to_csv(emissions_csv, index=False)
        print(f"[runner] Updated {len(updates)} rows in {emissions_csv}")

    return pd.read_csv(emissions_csv)
