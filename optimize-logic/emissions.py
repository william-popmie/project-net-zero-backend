import subprocess
import sys
import tempfile
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHMARK_TEMPLATE = """\
import sys
sys.path.insert(0, {optimize_logic_path!r})
from codecarbon import EmissionsTracker

{function_source}

tracker = EmissionsTracker(save_to_file=False, measure_power_secs=1, log_level="ERROR")
tracker.start()
for _ in range({iterations}):
    {benchmark_call}
emissions = tracker.stop()
print(f"EMISSIONS:{{emissions}}")
"""


def measure_emissions_for_source(
    function_source: str,
    function_name: str,
    benchmark_call: str | None = None,
    iterations: int = 10_000,
) -> float:
    if benchmark_call is None:
        benchmark_call = f"{function_name}(97)"

    optimize_logic_path = os.path.join(PROJECT_ROOT, "optimize-logic")

    script = BENCHMARK_TEMPLATE.format(
        optimize_logic_path=optimize_logic_path,
        function_source=function_source,
        iterations=iterations,
        benchmark_call=benchmark_call,
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
