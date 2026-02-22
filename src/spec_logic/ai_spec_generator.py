from __future__ import annotations

import ast
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer_logic import llm_client


def _extract_code(raw: str) -> str:
    """Pull the first fenced code block, or trim any prose preamble and return the code."""
    # Prefer an explicit fenced block
    match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fence: drop any leading prose and start from the first Python-looking line
    lines = raw.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ", "#")):
            return "\n".join(lines[i:]).strip()

    return raw.strip()


def generate_spec(
    function_id: str,
    function_source: str,
    file_path: str,
    engine: str = "claude",
) -> str:
    """Generate a concise pytest spec for one function via the requested engine.

    Returns:
        Valid Python test code as a string.

    Raises:
        ValueError: if the response is empty or returned code is not valid Python.
    """
    prompt = f"""Write a short pytest test file for the function below. Keep it under 40 lines.

Function: {function_id}
File: {file_path}

```python
{function_source}
```

Requirements:
- 2-3 test functions covering the main case and one edge case
- Import the function from its module path based on the source file
- pytest style (assert, no unittest)
- No setup/teardown boilerplate
- Output ONLY raw Python, no markdown, no commentary"""

    last_err: Exception | None = None

    for attempt in range(1, 2):  # up to 1 attempts
        raw = llm_client.generate_spec(prompt, engine=engine)
        code = _extract_code(raw)
        if not code:
            last_err = ValueError(f"{engine} returned empty response")
            continue
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            last_err = ValueError(f"Syntax error in generated code: {e}")
            if attempt < 3:
                print(f"  [generate] syntax error on attempt {attempt}, retrying...")

    raise last_err
