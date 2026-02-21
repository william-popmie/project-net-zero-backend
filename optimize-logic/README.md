# optimize-logic

Takes a Python function and returns a more carbon-efficient version of it.

---

## How it works

The pipeline is orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph) and loops until it finds an optimized version that both passes tests and emits less CO₂ than the original, or exhausts the attempt budget (max 4).

```
START
  │
  ▼
measure_baseline          ← CodeCarbon measures original function
  │
  ▼
optimize ◄────────────────────────────────┐
  │                                       │
  ▼                                       │
run_tests                                 │
  │                                       │
  ├─ FAIL + attempts left ────────────────┘
  │
  ├─ FAIL + attempts exhausted ──► END
  │
  ▼
measure_emissions         ← CodeCarbon measures optimized function
  │
  ├─ emissions improved ──► save_output ──► END
  │
  ├─ not improved + attempts left ────────► optimize (retry)
  │
  └─ not improved + attempts exhausted ──► END
```

---

## Interface

### Input — `FunctionSpec`

```python
from function_spec import FunctionSpec

spec = FunctionSpec(
    function_name="my_func",       # used for benchmark calls and output filename
    module_path="path/to/file.py", # where the function lives (for the caller to write back)
    function_source="def my_func(x):\n    ...",  # raw source of the function
    test_source="def test_my_func():\n    ...",  # pytest-style test stubs
)
```

### Output

```python
from optimizer import optimize_function

optimized_source: str = optimize_function(spec)
```

Returns the optimized function source as a plain string. Also writes it to `output-folder/{function_name}_optimized.py`.

---

## Configuration

### `.env`

```
ANTHROPIC_API_KEY=sk-ant-...
```

Place a `.env` file at the project root — it is loaded automatically via `python-dotenv` when the optimizer runs:

### `.codecarbon.config`

CodeCarbon is pre-configured at the project root. The key settings:

| Setting | Value | Note |
|---|---|---|
| `tracking_mode` | `process` | No privileged hardware access needed |
| `measure_power_secs` | `15` | Default interval (overridden to `1` inside benchmark runs) |
| `save_to_file` | `true` | Main tracker writes to `output-folder/codecarbon_output/` |
| `offline` | `false` | Uses live carbon intensity data for Sweden |

Benchmark subprocess runs override `save_to_file=False` and `measure_power_secs=1` so they complete in ~2 s and don't pollute `emissions.csv`.

---

## Smoke test

```bash
python optimize-logic/optimizer.py
```

Runs the full pipeline on a hardcoded `is_prime` example and prints the optimized source.
