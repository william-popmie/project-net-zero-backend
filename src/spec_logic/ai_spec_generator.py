from __future__ import annotations

import ast
import os
import re

import anthropic

MODEL = "claude-haiku-4-5-20251001"


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
) -> str:
    """Call Claude to generate a concise pytest spec for one function.

    Returns:
        Valid Python test code as a string.

    Raises:
        ValueError: if the API key is missing, response is empty,
                    or returned code is not valid Python.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

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

    client = anthropic.Anthropic(api_key=api_key)
    last_err: Exception | None = None

    for attempt in range(1, 4):  # up to 3 attempts
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        code = _extract_code(response.content[0].text)
        if not code:
            last_err = ValueError("Claude returned empty response")
            continue
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            last_err = ValueError(f"Syntax error in generated code: {e}")
            if attempt < 3:
                print(f"  [generate] syntax error on attempt {attempt}, retrying...")

    raise last_err
