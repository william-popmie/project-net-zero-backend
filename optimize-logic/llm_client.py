"""
llm_client.py — Unified LLM client for the optimizer pipeline.

All model-switching logic lives here. Callers pass an `engine` string;
this module dispatches to the appropriate backend (Claude or Crusoe).
"""

import re
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert Python performance engineer. "
    "When asked to optimize a function, return ONLY the optimized function "
    "inside a single ```python code block. No explanations, no other text."
)


# ── Public entry-points ───────────────────────────────────────────────────────

def rewrite(source_code: str, test_code: str, last_test_output: str, engine: str) -> dict:
    """Return optimized source code using the requested engine.

    Returns:
        {
            "rewritten_code": str,
            "duration_seconds": float,
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
        }
    """
    if engine == "crusoe":
        return _rewrite_crusoe(source_code, test_code, last_test_output)
    return _rewrite_claude(source_code, test_code, last_test_output)


def generate_spec(prompt: str, engine: str) -> str:
    """Generate a test spec / docstring for a function using the requested engine.

    Returns the raw text response from the model.
    """
    if engine == "crusoe":
        return _generate_spec_crusoe(prompt)
    return _generate_spec_claude(prompt)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_optimize_user_content(source_code: str, test_code: str, last_test_output: str) -> str:
    user_content = (
        f"Optimize the following Python function to use less CPU and memory "
        f"(and therefore less energy/carbon emissions). "
        f"Return ONLY the optimized function inside a ```python code block. "
        f"Do not include any explanation.\n\n"
        f"```python\n{source_code}\n```"
    )
    if last_test_output and test_code:
        user_content += (
            f"\n\nThe previous version failed the tests. Here is the output:\n"
            f"```\n{last_test_output}\n```\n"
            f"Make sure the optimized function still passes these tests:\n"
            f"```python\n{test_code}\n```"
        )
    return user_content


def _rewrite_claude(source_code: str, test_code: str, last_test_output: str) -> dict:
    client = anthropic.Anthropic()
    user_content = _build_optimize_user_content(source_code, test_code, last_test_output)

    start = time.time()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    duration = time.time() - start

    raw = message.content[0].text
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    rewritten = match.group(1).strip() if match else raw.strip()

    return {
        "rewritten_code": rewritten,
        "duration_seconds": round(duration, 3),
        "usage": {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        },
    }


def _rewrite_crusoe(source_code: str, test_code: str, last_test_output: str) -> dict:
    import crusoe_client
    return crusoe_client.rewrite(source_code, test_code, last_test_output)


def _generate_spec_claude(prompt: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _generate_spec_crusoe(prompt: str) -> str:
    import crusoe_client
    from openai import OpenAI
    import os

    api_key = os.getenv("CRUSOE_API_KEY", "")
    if not api_key:
        raise ValueError("CRUSOE_API_KEY must be set in .env or environment")

    client = OpenAI(api_key=api_key, base_url="https://api.crusoe.ai/v1")
    response = client.chat.completions.create(
        model=crusoe_client.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()
