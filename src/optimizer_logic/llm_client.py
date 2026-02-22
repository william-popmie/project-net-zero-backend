"""
llm_client.py — Model-agnostic dispatch layer.

Callers own their own prompts; this module handles routing to the right
backend (Claude or Crusoe) and returns the raw text response.

Supported engines:
    "claude"  — Anthropic Claude (default)
    "crusoe"  — Crusoe inference API (requires CRUSOE_API_KEY in .env)
"""

from __future__ import annotations

import time

import anthropic
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_model(
    system: str,
    user: str,
    engine: str = "claude",
    max_tokens: int = 4096,
) -> tuple[str, float, dict]:
    """Send a system+user prompt to the requested engine.

    Returns:
        (text, duration_seconds, usage)
        usage = {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    if engine == "crusoe":
        return _call_crusoe(system, user, max_tokens)
    return _call_claude(system, user, max_tokens)


def generate_spec(prompt: str, engine: str = "claude") -> str:
    """Generate a test spec for a function using the requested engine.

    Returns the raw text response from the model.
    """
    text, _, _ = call_model(system="", user=prompt, engine=engine, max_tokens=2048)
    return text


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _call_claude(system: str, user: str, max_tokens: int) -> tuple[str, float, dict]:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user}]
    kwargs: dict = dict(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    start = time.time()
    message = client.messages.create(**kwargs)
    duration = round(time.time() - start, 3)

    text = message.content[0].text
    usage = {
        "prompt_tokens":     message.usage.input_tokens,
        "completion_tokens": message.usage.output_tokens,
        "total_tokens":      message.usage.input_tokens + message.usage.output_tokens,
    }
    return text, duration, usage


def _call_crusoe(system: str, user: str, max_tokens: int) -> tuple[str, float, dict]:
    from . import crusoe_client
    import os
    from openai import OpenAI

    api_key = os.getenv("CRUSOE_API_KEY", "")
    if not api_key:
        raise ValueError("CRUSOE_API_KEY must be set in .env or environment")

    oc = OpenAI(api_key=api_key, base_url="https://api.crusoe.ai/v1")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    start = time.time()
    response = oc.chat.completions.create(
        model=crusoe_client.DEFAULT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    duration = round(time.time() - start, 3)

    text = response.choices[0].message.content.strip()
    usage_obj = response.usage
    usage = {
        "prompt_tokens":     usage_obj.prompt_tokens if usage_obj else 0,
        "completion_tokens": usage_obj.completion_tokens if usage_obj else 0,
        "total_tokens":      usage_obj.total_tokens if usage_obj else 0,
    }
    return text, duration, usage
