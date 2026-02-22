"""
venv_runner.py — isolated venv creation, project layout, spec running, and
emissions measurement for the optimizer pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Measure script template (written to tmp/ and run inside venv)
# ---------------------------------------------------------------------------

MEASURE_SCRIPT = """\
from codecarbon import EmissionsTracker
import pytest

tracker = EmissionsTracker(measure_power_secs=0.1, log_level="ERROR", save_to_file=False)
tracker.start()
exit_code = pytest.main(["tests/", "-v", "--tb=short"])
emissions = tracker.stop()
print(f"EMISSIONS:{emissions}")
print(f"EXIT_CODE:{exit_code}")
"""

PYTEST_INI = """\
[pytest]
testpaths = tests
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_shared_venv(project_root: str, venv_dir: Path) -> tuple[Path, Path]:
    """Create a venv at *venv_dir*, install project requirements + pytest + codecarbon.

    Returns (pip_path, python_path).
    """
    project_root_path = Path(project_root)

    # Collect project requirements
    req_packages: list[str] = []
    for candidate in [
        project_root_path / "requirements.txt",
        project_root_path.parent / "requirements.txt",
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith("pip "):
                    continue
                req_packages.append(line)
            break

    # Always include testing / measurement deps
    for pkg in ("pytest", "codecarbon"):
        if not any(p.lower().startswith(pkg) for p in req_packages):
            req_packages.append(pkg)

    print(f"[venv_runner] Creating venv at {venv_dir} ...")
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
    )

    pip = _venv_bin(venv_dir, "pip")
    python = _venv_bin(venv_dir, "python")

    if req_packages:
        print(f"[venv_runner] Installing {len(req_packages)} packages: {req_packages}")
        subprocess.run(
            [str(pip), "install", "-q", *req_packages],
            check=True,
        )

    return pip, python


def build_project_dir(tmp: Path, func_record: dict, source_content: str) -> None:
    """Write a self-contained project layout into *tmp*:

        tmp/
        ├── pytest.ini
        ├── src/
        │   ├── __init__.py
        │   └── <module>.py    ← source_content (full original source file)
        └── tests/
            ├── __init__.py
            └── test_spec.py   ← func_record["spec_code"]
    """
    # Determine target path from func_record["file"] (e.g. "src/activations.py")
    relative_file = Path(func_record["file"])  # e.g. src/activations.py

    target = tmp / relative_file
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write __init__.py for every package level under tmp/
    pkg = target.parent
    while pkg != tmp:
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text("")
        pkg = pkg.parent

    target.write_text(source_content)

    # Write tests/
    tests_dir = tmp / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_spec.py").write_text(func_record["spec_code"])

    # Write pytest.ini
    (tmp / "pytest.ini").write_text(PYTEST_INI)


def run_spec(python: Path, tmp: Path) -> tuple[bool, str]:
    """Run pytest without emissions measurement.  Returns (passed, combined_output)."""
    result = subprocess.run(
        [str(python), "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(tmp),
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def measure_emissions_via_pytest(
    python: Path,
    tmp: Path,
    runs: int = 1,
) -> tuple[float, bool]:
    """Run measure.py inside venv *runs* times and return the average kg CO2eq.

    measure.py wraps pytest.main() in CodeCarbon (in-process).
    Returns (avg_emissions_kg, all_passed).
    Returns (0.0, False) if any run fails to parse emissions.
    """
    measure_py = tmp / "measure.py"
    measure_py.write_text(MEASURE_SCRIPT)

    emissions_values: list[float] = []
    all_passed = True

    for i in range(runs):
        result = subprocess.run(
            [str(python), "measure.py"],
            capture_output=True,
            text=True,
            cwd=str(tmp),
        )
        combined = result.stdout + result.stderr

        emissions: Optional[float] = None
        exit_code: Optional[int] = None

        for line in combined.splitlines():
            if line.startswith("EMISSIONS:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    val = float(raw)
                    emissions = val
                except ValueError:
                    pass
            elif line.startswith("EXIT_CODE:"):
                try:
                    exit_code = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        if emissions is None or exit_code is None:
            print(f"[venv_runner] run {i+1}: failed to parse output — {combined[:400]}")
            return 0.0, False

        if exit_code != 0:
            all_passed = False

        emissions_values.append(emissions)

    if not emissions_values:
        return 0.0, False

    avg = sum(emissions_values) / len(emissions_values)
    return avg, all_passed


def replace_function_in_source(
    full_source: str,
    new_code: str,
    start_line: int,
    end_line: int,
) -> str:
    """Splice *new_code* into *full_source* replacing lines [start_line..end_line] (1-indexed, inclusive)."""
    lines = full_source.splitlines(keepends=True)
    before = lines[: start_line - 1]
    after = lines[end_line:]
    # Ensure new_code ends with a newline
    replacement = new_code if new_code.endswith("\n") else new_code + "\n"
    return "".join(before) + replacement + "".join(after)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _venv_bin(venv_dir: Path, name: str) -> Path:
    """Return path to a binary inside the venv, cross-platform."""
    # Unix
    candidate = venv_dir / "bin" / name
    if candidate.exists():
        return candidate
    # Windows
    candidate = venv_dir / "Scripts" / (name + ".exe")
    if candidate.exists():
        return candidate
    # Fallback (let OS resolve)
    return venv_dir / "bin" / name
