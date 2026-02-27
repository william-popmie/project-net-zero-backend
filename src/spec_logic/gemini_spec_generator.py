from __future__ import annotations

import ast
import os
import re
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

GEMINI_MODEL = "gemini-3-flash-preview"
RETRY_DELAYS = [2.0, 5.0, 15.0]
MAX_ATTEMPTS = 3


def _extract_code(raw: str) -> str:
    """Pull the first fenced code block, or trim any prose preamble and return the code."""
    match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

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
    """Call Gemini to generate a concise pytest spec for one function.

    Returns:
        Valid Python test code as a string.

    Raises:
        ValueError: if the API key is missing, response is empty,
                    or returned code is not valid Python after all retries.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(f"GEMINI_API_KEY not set (function: {function_id})")

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

    client = genai.Client(api_key=api_key)
    last_err: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=1024),
            )
            code = _extract_code(response.text or "")
            if not code:
                raise ValueError(f"Gemini returned empty response for {function_id}")
            ast.parse(code)
            return code
        except (ClientError, ServerError) as e:
            last_err = e
            if attempt < MAX_ATTEMPTS:
                delay = RETRY_DELAYS[attempt - 1]
                print(
                    f"  [gemini] API error on attempt {attempt} for {function_id}, "
                    f"retrying in {delay}s: {e}",
                    flush=True,
                )
                time.sleep(delay)
        except SyntaxError as e:
            last_err = ValueError(f"Syntax error on attempt {attempt} for {function_id}: {e}")
            if attempt < MAX_ATTEMPTS:
                print(
                    f"  [gemini] syntax error on attempt {attempt} for {function_id}, retrying...",
                    flush=True,
                )

    raise last_err
