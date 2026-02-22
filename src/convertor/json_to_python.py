"""Convert optimizer results.json to reconstructed Python source files."""

from __future__ import annotations

import ast
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cost-estimation constants  (used in results.json summary)
# ---------------------------------------------------------------------------
# We derive electricity from measured CO₂ using average grid carbon intensity,
# then apply two cost lenses companies care about:
#
#   1. Operational / electricity cost  — what you actually pay on the bill
#   2. Carbon credit value             — regulatory / ESG exposure (EU ETS)
#
# IEA World Energy Outlook 2023 global average: 0.233 kg CO₂ / kWh
_CARBON_INTENSITY_KG_PER_KWH: float = 0.233
# US industrial electricity price (EIA 2024 avg): $0.084/kWh
# EU industrial electricity price (Eurostat H1-2024): ~$0.155/kWh
# We use a blended global estimate of $0.10/kWh — conservative midpoint
_ELECTRICITY_USD_PER_KWH: float = 0.10
# EU ETS carbon allowance price (Q1 2025 avg): ~$65 / tonne CO₂
_EU_ETS_USD_PER_TONNE: float = 65.0
# Voluntary Carbon Market (VCM) mid-range: ~$15 / tonne CO₂
_VCM_USD_PER_TONNE: float = 15.0
# Assumed production scale for annual projections
_ANNUAL_TRAINING_RUNS: int = 1_000


def _reindent(code: str, target_indent: str) -> str:
    """Re-indent a code block so its first non-empty line uses target_indent."""
    lines = code.split('\n')

    # Detect the current base indentation from the first non-empty line
    base_indent = ''
    for line in lines:
        if line.strip():
            base_indent = line[: len(line) - len(line.lstrip())]
            break

    base_len = len(base_indent)
    result = []
    for line in lines:
        if not line.strip():
            result.append('')
        elif len(line) >= base_len and line[:base_len] == base_indent:
            result.append(target_indent + line[base_len:])
        else:
            result.append(target_indent + line.lstrip())
    return '\n'.join(result)


def _find_function_line_range(source: str, qualified_name: str) -> tuple[int, int] | None:
    """
    Return (start_line, end_line) 1-indexed inclusive for a function/method.
    qualified_name may be 'func_name' or 'ClassName.method_name'.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    parts = qualified_name.split('.')

    if len(parts) == 1:
        func_name = parts[0]
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
            ):
                return node.lineno, node.end_lineno

    elif len(parts) == 2:
        class_name, method_name = parts
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.walk(node):
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == method_name
                    ):
                        return child.lineno, child.end_lineno

    return None


def _build_results_summary(json_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build a results.json summary dict from optimizer output.

    Uses actual CodeCarbon-measured emissions where available, then derives
    electricity and cost estimates from well-sourced constants.
    """
    functions = json_data.get("functions", [])
    total = len(functions)
    optimized = [f for f in functions if f.get("success")]
    skipped = [f for f in functions if not f.get("success")]

    # ── Measured CO₂ figures (only functions with real measurements) ──────────
    measured = [
        f for f in optimized
        if f.get("baseline_emissions_kg") is not None
        and f.get("optimized_emissions_kg") is not None
    ]

    total_baseline_kg = sum(f["baseline_emissions_kg"] for f in measured)
    total_optimized_kg = sum(f["optimized_emissions_kg"] for f in measured)
    total_saved_kg = total_baseline_kg - total_optimized_kg

    reductions = [f["reduction_pct"] for f in measured if f.get("reduction_pct") is not None]
    avg_reduction_pct = sum(reductions) / len(reductions) if reductions else 0.0
    max_reduction = max(reductions, default=0.0)
    min_reduction = min(reductions, default=0.0)

    files_optimized = sorted({f["file"] for f in optimized})

    # ── Derive energy saved from measured CO₂ ─────────────────────────────────
    energy_saved_kwh = total_saved_kg / _CARBON_INTENSITY_KG_PER_KWH if total_saved_kg > 0 else 0.0

    # ── Cost estimates ─────────────────────────────────────────────────────────
    electricity_cost_saved_usd = energy_saved_kwh * _ELECTRICITY_USD_PER_KWH
    eu_ets_value_usd = (total_saved_kg / 1000) * _EU_ETS_USD_PER_TONNE
    vcm_value_usd = (total_saved_kg / 1000) * _VCM_USD_PER_TONNE

    # ── Annual projection (scale up by assumed production runs) ───────────────
    annual_saved_kg = total_saved_kg * _ANNUAL_TRAINING_RUNS
    annual_energy_kwh = energy_saved_kwh * _ANNUAL_TRAINING_RUNS
    annual_electricity_usd = electricity_cost_saved_usd * _ANNUAL_TRAINING_RUNS
    annual_eu_ets_usd = eu_ets_value_usd * _ANNUAL_TRAINING_RUNS
    annual_vcm_usd = vcm_value_usd * _ANNUAL_TRAINING_RUNS

    # ── Per-function breakdown ─────────────────────────────────────────────────
    per_function = []
    for f in optimized:
        b = f.get("baseline_emissions_kg")
        o = f.get("optimized_emissions_kg")
        saved = (b - o) if (b is not None and o is not None) else None
        per_function.append({
            "id": f["id"],
            "name": f["name"],
            "file": f["file"],
            "baseline_emissions_kg": b,
            "optimized_emissions_kg": o,
            "co2_saved_kg": saved,
            "reduction_pct": f.get("reduction_pct", 0.0),
            "optimization_attempts": f.get("optimization_attempts", 0),
        })
    per_function.sort(key=lambda x: x["reduction_pct"], reverse=True)

    skipped_summary = [
        {"id": f["id"], "name": f["name"], "file": f["file"], "reason": f.get("skip_reason")}
        for f in skipped
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": json_data.get("project_root"),
        "summary": {
            "total_functions_analyzed": total,
            "functions_optimized": len(optimized),
            "functions_skipped": len(skipped),
            "success_rate_pct": round(len(optimized) / total * 100, 1) if total else 0.0,
            "files_optimized": files_optimized,
            "functions_with_measurements": len(measured),
            "avg_co2_reduction_pct": round(avg_reduction_pct, 2),
            "max_co2_reduction_pct": round(max_reduction, 2),
            "min_co2_reduction_pct": round(min_reduction, 2),
            "total_baseline_emissions_kg": total_baseline_kg,
            "total_optimized_emissions_kg": total_optimized_kg,
            "total_co2_saved_kg": total_saved_kg,
            "total_co2_saved_g": total_saved_kg * 1000,
        },
        "cost_estimates": {
            "_methodology": (
                "CO₂ → energy via global avg grid intensity (IEA 2023). "
                "Energy → electricity cost at blended global industrial rate. "
                "Carbon credit value at EU ETS Q1-2025 and VCM mid-range prices. "
                "Annual projection assumes the given number of training runs/year. "
                "All figures are estimates — actual savings depend on grid mix and usage patterns."
            ),
            "assumptions": {
                "carbon_intensity_kg_per_kwh": _CARBON_INTENSITY_KG_PER_KWH,
                "electricity_price_usd_per_kwh": _ELECTRICITY_USD_PER_KWH,
                "eu_ets_carbon_price_usd_per_tonne": _EU_ETS_USD_PER_TONNE,
                "vcm_carbon_price_usd_per_tonne": _VCM_USD_PER_TONNE,
                "annual_training_runs_assumed": _ANNUAL_TRAINING_RUNS,
            },
            "per_benchmark_run": {
                "energy_saved_kwh": energy_saved_kwh,
                "electricity_cost_saved_usd": electricity_cost_saved_usd,
                "eu_ets_carbon_credit_value_usd": eu_ets_value_usd,
                "vcm_carbon_credit_value_usd": vcm_value_usd,
            },
            "annual_projection": {
                "co2_saved_kg": annual_saved_kg,
                "energy_saved_kwh": annual_energy_kwh,
                "electricity_cost_saved_usd": annual_electricity_usd,
                "eu_ets_carbon_credit_value_usd": annual_eu_ets_usd,
                "vcm_carbon_credit_value_usd": annual_vcm_usd,
            },
        },
        "optimized_functions": per_function,
        "skipped_functions": skipped_summary,
    }


def convert_and_write(
    json_data: dict[str, Any],
    input_repo_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """
    Copy input_repo_dir → output_dir, then splice optimized functions in-place.

    Returns a dict mapping output file paths to their final source code.
    """
    project_root = Path(json_data["project_root"])

    # ── Step 1: Copy entire input-repo to output-repo ────────────────────────
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_repo_dir, output_dir, ignore=shutil.ignore_patterns('.git'))
    print(f"[convertor] Copied {input_repo_dir} → {output_dir}")

    # ── Step 2: Group successful optimizations by source file ─────────────────
    by_file: dict[str, list[dict]] = defaultdict(list)
    for func in json_data.get("functions", []):
        if func.get("success"):
            by_file[func["file"]].append(func)

    written: dict[str, str] = {}

    # ── Step 3: Splice optimized functions into the copied files ──────────────
    for rel_file, funcs in by_file.items():
        # rel_file is relative to project_root (e.g. "src/activations.py")
        src_file = project_root / rel_file

        # Compute the path relative to input_repo_dir so we can mirror it
        try:
            rel_from_input = src_file.relative_to(input_repo_dir)
        except ValueError:
            try:
                project_rel = project_root.relative_to(input_repo_dir)
                rel_from_input = project_rel / rel_file
            except ValueError:
                print(f"[WARN] Cannot determine output path for {rel_file}, skipping.")
                continue

        dest_file = output_dir / rel_from_input
        if not dest_file.exists():
            print(f"[WARN] Expected output file not found: {dest_file}, skipping.")
            continue

        source = dest_file.read_text(encoding='utf-8')
        source_lines = source.splitlines(keepends=True)

        # Locate every function's line range in the current source
        func_ranges: list[tuple[int, int, dict]] = []
        for func in funcs:
            rng = _find_function_line_range(source, func["qualified_name"])
            if rng is None:
                print(f"[WARN] Could not locate {func['qualified_name']} in {rel_file}, skipping.")
                continue
            func_ranges.append((*rng, func))

        # Replace in reverse line order so earlier replacements don't shift later ones
        func_ranges.sort(key=lambda x: x[0], reverse=True)

        for start_line, end_line, func in func_ranges:
            # Preserve the original indentation of the def line
            original_def_line = source_lines[start_line - 1]
            target_indent = original_def_line[: len(original_def_line) - len(original_def_line.lstrip())]

            optimized = _reindent(func["optimized_function_code"], target_indent)
            optimized_lines = [line + '\n' for line in optimized.splitlines()]

            source_lines[start_line - 1 : end_line] = optimized_lines
            print(f"[OK] {func['qualified_name']} → {rel_from_input}")

        final_source = ''.join(source_lines)
        dest_file.write_text(final_source, encoding='utf-8')
        written[str(dest_file)] = final_source

    # ── Step 4: Write results.json summary to output-repo root ───────────────
    summary = _build_results_summary(json_data)
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[convertor] Results summary → {results_path}")

    return written


def write_python_files(
    json_file: Path | str,
    output_dir: Path | str,
    input_repo_dir: Path | str,
) -> dict[str, str]:
    """
    Read optimizer result.json, mirror input_repo_dir into output_dir, and
    splice in all successfully optimized functions.

    Args:
        json_file:      Path to optimizer result.json.
        output_dir:     Destination directory (output-repo).
        input_repo_dir: Top-level input-repo directory whose structure to mirror.

    Returns:
        Dict mapping written file paths to their final source code.
    """
    json_data = json.loads(Path(json_file).read_text(encoding='utf-8'))
    return convert_and_write(
        json_data=json_data,
        input_repo_dir=Path(input_repo_dir).resolve(),
        output_dir=Path(output_dir).resolve(),
    )
