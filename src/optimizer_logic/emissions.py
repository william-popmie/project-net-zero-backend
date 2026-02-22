import ast
import subprocess
import sys
import tempfile
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARK_TEMPLATE = """\
from __future__ import annotations
import sys
import os
import typing
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple, Type, Union
try:
    from typing import Annotated
except ImportError:
    pass
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, OrderedDict, Counter, deque
try:
    import numpy as np
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
sys.path.insert(0, {optimize_logic_path!r})
from codecarbon import EmissionsTracker

{function_source}

os.makedirs({codecarbon_output_dir!r}, exist_ok=True)
tracker = EmissionsTracker(measure_power_secs=1, log_level="ERROR", output_dir={codecarbon_output_dir!r})
tracker.start()
try:
    for _ in range({iterations}):
        {benchmark_call}
except Exception:
    pass
emissions = tracker.stop()
if emissions is None:
    emissions = 0.0
print(f"EMISSIONS:{{emissions}}")
"""


def measure_emissions_for_source(
    function_source: str,
    function_name: str,
    benchmark_call: str | None = None,
    iterations: int = 10_000,
) -> float:
    if benchmark_call is None:
        try:
            tree = ast.parse(function_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    num_args = len(node.args.args)
                    args = ", ".join(["97"] * num_args)
                    benchmark_call = f"{function_name}({args})"
                    break
        except Exception:
            pass
        if benchmark_call is None:
            benchmark_call = f"{function_name}(97)"

    optimize_logic_path = os.path.join(PROJECT_ROOT, "src", "optimizer_logic")
    codecarbon_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "codecarbon_output")
    os.makedirs(codecarbon_output_dir, exist_ok=True)

    script = BENCHMARK_TEMPLATE.format(
        optimize_logic_path=optimize_logic_path,
        function_source=function_source,
        iterations=iterations,
        benchmark_call=benchmark_call,
        codecarbon_output_dir=codecarbon_output_dir,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        for line in result.stdout.splitlines():
            if line.startswith("EMISSIONS:"):
                return float(line.split(":", 1)[1])
        raise RuntimeError(
            f"No EMISSIONS line found.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    finally:
        os.unlink(tmp_path)
