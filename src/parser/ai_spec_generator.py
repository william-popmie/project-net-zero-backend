from __future__ import annotations

import ast
from pathlib import Path

from optimizer_logic import llm_client


def determine_test_file_path(project_root: Path, source_file_path: str) -> Path:
    """
    Determine where the test file should be placed.

    Convention: src/app/math_utils.py -> tests/app/test_math_utils.py
    """
    source_path = Path(source_file_path)

    # Remove common source prefixes (src/, lib/, app/)
    parts = list(source_path.parts)
    if parts and parts[0] in ("src", "lib", "app"):
        parts = parts[1:]

    # Build test path: tests/<subpath>/test_<module>.py
    filename_stem = Path(parts[-1]).stem
    test_filename = f"test_{filename_stem}.py"

    test_dir = project_root / "tests"
    if len(parts) > 1:
        test_dir = test_dir / Path(*parts[:-1])

    return test_dir / test_filename


def generate_tests(
    function_id: str,
    function_code: str,
    source_file: str,
    project_root: Path,
    engine: str = "claude",
) -> str:
    """
    Generate pytest code for a function using the requested engine.

    Args:
        function_id: Dotted qualified name, e.g. "src.app.math_utils.add"
        function_code: The actual function source code
        source_file: Relative path, e.g. "src/app/math_utils.py"
        project_root: Root of the project (for context)
        engine: "claude" or "crusoe"

    Returns:
        Generated pytest code as a string (valid Python).

    Raises:
        ValueError: If generation fails or code is not valid Python.
    """
    prompt = f"""Generate a complete, production-ready pytest test for this Python function.

Function: {function_id}
Source file: {source_file}

```python
{function_code}
```

Requirements:
- Use pytest conventions (test_function_name)
- Import the function correctly based on the file path
- Cover main use cases and edge cases
- Use clear, descriptive test names
- Use assert statements, not unittest style
- Make tests independent and deterministic

Respond with ONLY the Python test code (no markdown, no explanations).
Include all necessary imports at the top."""

    test_code = llm_client.generate_spec(prompt=prompt, engine=engine).strip()

    # Remove markdown code fences if present
    if test_code.startswith("```"):
        lines = test_code.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("```"):
                start_idx = i + 1
                break
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith("```"):
                end_idx = i
                break
        test_code = "\n".join(lines[start_idx:end_idx]).strip()

    if not test_code:
        raise ValueError("Model returned empty test code")

    # Validate that the returned code is syntactically valid Python
    try:
        ast.parse(test_code)
    except SyntaxError as e:
        raise ValueError(
            f"Generated test code is not valid Python: {e}\nCode:\n{test_code}"
        )

    return test_code
