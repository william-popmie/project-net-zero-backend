from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

CACHE_DIR = Path.home() / ".cache" / "function_spec_graph"


@dataclass
class MatchResult:
    matches: bool
    confidence_score: float
    reasoning: str
    source: str  # "ai" or "heuristic"


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(test_id: str, source_id: str) -> str:
    return f"{test_id}___{source_id}".replace("/", "_").replace(".", "_")


def get_cached_result(test_id: str, source_id: str) -> MatchResult | None:
    ensure_cache_dir()
    cache_file = CACHE_DIR / f"{get_cache_key(test_id, source_id)}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return MatchResult(**data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def save_cache_result(test_id: str, source_id: str, result: MatchResult) -> None:
    ensure_cache_dir()
    cache_file = CACHE_DIR / f"{get_cache_key(test_id, source_id)}.json"
    cache_file.write_text(json.dumps(vars(result), indent=2))


def ai_match_function(
    test_qualified_name: str,
    test_code_snippet: str,
    source_qualified_name: str,
    source_code_snippet: str,
) -> MatchResult:
    """
    Use Claude to determine if a test validates a source function.
    
    Args:
        test_qualified_name: e.g., "tests.test_math.test_add"
        test_code_snippet: the actual test function code
        source_qualified_name: e.g., "src.app.math_utils.add"
        source_code_snippet: the actual source function code
    
    Returns:
        MatchResult with confidence score and reasoning
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it or use --use-heuristic flag."
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze these two Python functions and determine if the test function validates the source function.

Test Function: {test_qualified_name}
```python
{test_code_snippet}
```

Source Function: {source_qualified_name}
```python
{source_code_snippet}
```

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "matches": true/false,
  "confidence_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Consider:
- Does the test call this function (directly or indirectly)?
- Does the test verify the function's behavior?
- Is the test logically related to this function?

Be strict: only return true if there's a clear relationship."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = response.content[0].text.strip()
        result_data = json.loads(response_text)
        return MatchResult(
            matches=result_data["matches"],
            confidence_score=float(result_data["confidence_score"]),
            reasoning=result_data["reasoning"],
            source="ai",
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise ValueError(f"Failed to parse Claude response: {e}\n{response_text}")


def match_with_ai(
    test_qualified_name: str,
    test_code_snippet: str,
    source_qualified_name: str,
    source_code_snippet: str,
) -> MatchResult:
    """Wrapper with caching."""
    cached = get_cached_result(test_qualified_name, source_qualified_name)
    if cached:
        return cached

    result = ai_match_function(
        test_qualified_name, test_code_snippet,
        source_qualified_name, source_code_snippet
    )
    save_cache_result(test_qualified_name, source_qualified_name, result)
    return result
