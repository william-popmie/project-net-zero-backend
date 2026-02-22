"""
crusoe_client.py — Thin OpenAI-compatible client for Crusoe's Inference API.

Used as an alternative rewrite engine alongside Claude in the optimizer graph.
"""

import os
import re
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CRUSOE_API_KEY = os.getenv("CRUSOE_API_KEY", "")
CRUSOE_BASE_URL = "https://api.crusoe.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are an expert Python performance engineer. "
    "When asked to optimize a function, return ONLY the optimized function "
    "inside a single ```python code block. No explanations, no other text."
)


def _get_client() -> OpenAI:
    if not CRUSOE_API_KEY:
        raise ValueError("CRUSOE_API_KEY must be set in .env or environment")
    return OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)


def rewrite(
    source_code: str,
    test_code: str = "",
    last_test_output: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> dict:
    """
    Send a Python function to Crusoe for energy-efficient rewriting.

    Returns:
        {
            "rewritten_code": str,
            "model": str,
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "duration_seconds": float,
        }
    """
    client = _get_client()

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

    last_error = None
    for attempt in range(3):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            duration = time.time() - start
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"Crusoe inference failed after 3 attempts: {last_error}")

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    rewritten = match.group(1).strip() if match else raw.strip()

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "rewritten_code": rewritten,
        "model": model,
        "usage": usage,
        "duration_seconds": round(duration, 3),
    }
