"""
benchmark.py — Run both Claude and Crusoe engines on test functions,
collect metrics, and generate a markdown comparison report.

Usage:
    python benchmark.py
"""

import os
import sys
import time
from datetime import datetime

from function_spec import FunctionSpec
from optimizer import run_engine

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output-folder")

# ── Test Functions ────────────────────────────────────────────────────────────

TEST_FUNCTIONS = [
    # ── Functions from pytorch-geoguessr.py ──────────────────────────────────
    FunctionSpec(
        function_name="extract_coordinates",
        module_path="pytorch-geoguessr.py",
        function_source="""\
import re

def extract_coordinates(filename):
    pattern = r'(\\d+)_(-?\\d+\\.\\d+)_(-?\\d+\\.\\d+)'
    match = re.search(pattern, filename)
    if match:
        timestamp, lat, lon = match.groups()
        return float(timestamp), float(lat), float(lon)
    return None, None, None""",
        test_source="""\
def test_extract_coordinates_valid():
    ts, lat, lon = extract_coordinates("1234567890_52.3676_4.9041.jpg")
    assert ts == 1234567890.0
    assert abs(lat - 52.3676) < 1e-4
    assert abs(lon - 4.9041) < 1e-4

def test_extract_coordinates_negative():
    ts, lat, lon = extract_coordinates("9999_-33.8688_151.2093.png")
    assert ts == 9999.0
    assert abs(lat - (-33.8688)) < 1e-4
    assert abs(lon - 151.2093) < 1e-4

def test_extract_coordinates_invalid():
    ts, lat, lon = extract_coordinates("no_coords_here.jpg")
    assert ts is None
    assert lat is None
    assert lon is None""",
        benchmark_call="extract_coordinates('1234567890_52.3676_4.9041.jpg')",
    ),
    FunctionSpec(
        function_name="slice_panorama",
        module_path="pytorch-geoguessr.py",
        function_source="""\
import numpy as np

def slice_panorama(panorama_img, num_slices=6):
    img = panorama_img
    width = img.shape[1]
    slice_width = width // num_slices
    slices = []
    for i in range(num_slices):
        start = i * slice_width
        end = (i + 1) * slice_width
        slice_img = img[:, start:end, :]
        slices.append(slice_img)
    return slices""",
        test_source="""\
import numpy as np

def test_slice_panorama_count():
    img = np.zeros((100, 600, 3), dtype=np.uint8)
    slices = slice_panorama(img, num_slices=6)
    assert len(slices) == 6

def test_slice_panorama_shape():
    img = np.zeros((100, 600, 3), dtype=np.uint8)
    slices = slice_panorama(img, num_slices=6)
    for s in slices:
        assert s.shape == (100, 100, 3)

def test_slice_panorama_content():
    img = np.arange(12).reshape(1, 4, 3)
    slices = slice_panorama(img, num_slices=2)
    assert len(slices) == 2
    assert np.array_equal(slices[0], img[:, :2, :])
    assert np.array_equal(slices[1], img[:, 2:4, :])

def test_slice_panorama_3_slices():
    img = np.ones((50, 300, 3), dtype=np.uint8)
    slices = slice_panorama(img, num_slices=3)
    assert len(slices) == 3
    for s in slices:
        assert s.shape == (50, 100, 3)""",
        benchmark_call="slice_panorama(np.zeros((100, 600, 3), dtype=np.uint8), 6)",
    ),
    FunctionSpec(
        function_name="augment_with_slices_logic",
        module_path="pytorch-geoguessr.py",
        function_source="""\
import os

def augment_with_slices_logic(file_list, num_slices=6):
    results = []
    for img_file in file_list:
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        base_name = os.path.splitext(img_file)[0]
        for i in range(num_slices):
            results.append(f"{base_name}_slice{i}.jpg")
    return results""",
        test_source="""\
def test_augment_basic():
    files = ["photo1.jpg", "photo2.png", "readme.txt"]
    result = augment_with_slices_logic(files, num_slices=6)
    assert len(result) == 12  # 2 images * 6 slices
    assert "photo1_slice0.jpg" in result
    assert "photo2_slice5.jpg" in result

def test_augment_no_images():
    files = ["readme.txt", "data.csv"]
    result = augment_with_slices_logic(files, num_slices=6)
    assert len(result) == 0

def test_augment_custom_slices():
    files = ["img.jpg"]
    result = augment_with_slices_logic(files, num_slices=3)
    assert len(result) == 3
    assert result == ["img_slice0.jpg", "img_slice1.jpg", "img_slice2.jpg"]

def test_augment_empty():
    result = augment_with_slices_logic([], num_slices=6)
    assert result == []""",
        benchmark_call="augment_with_slices_logic(['img1.jpg','img2.png','img3.jpeg','skip.txt'] * 50, 6)",
    ),
    FunctionSpec(
        function_name="ensemble_predict_sim",
        module_path="pytorch-geoguessr.py",
        function_source="""\
import torch
import torch.nn.functional as F

def ensemble_predict_sim(model_outputs):
    predictions = []
    for output in model_outputs:
        softmaxed = F.softmax(output, dim=1)
        predictions.append(softmaxed)
    avg_preds = torch.mean(torch.stack(predictions), dim=0)
    return avg_preds""",
        test_source="""\
import torch

def test_ensemble_predict_shape():
    outputs = [torch.randn(4, 10) for _ in range(3)]
    result = ensemble_predict_sim(outputs)
    assert result.shape == (4, 10)

def test_ensemble_predict_probabilities():
    outputs = [torch.randn(2, 5) for _ in range(3)]
    result = ensemble_predict_sim(outputs)
    sums = result.sum(dim=1)
    for s in sums:
        assert abs(s.item() - 1.0) < 1e-5

def test_ensemble_predict_single_model():
    outputs = [torch.tensor([[1.0, 2.0, 3.0]])]
    result = ensemble_predict_sim(outputs)
    expected = F.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=1)
    assert torch.allclose(result, expected, atol=1e-5)""",
        benchmark_call="ensemble_predict_sim([torch.randn(8, 20) for _ in range(5)])",
    ),
]


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark():
    """Run both engines on all test functions and collect results."""
    all_results = []

    for spec in TEST_FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {spec.function_name}")
        print(f"{'='*60}")

        func_results = {"function_name": spec.function_name}

        for idx, engine in enumerate(["claude", "crusoe"]):
            if idx > 0:
                time.sleep(2)  # avoid rate limits between engines
            print(f"\n--- Running {engine} engine ---")
            try:
                result = run_engine(spec, engine)
                func_results[engine] = result
                reduction = 0.0
                if result["baseline_emissions"] > 0:
                    reduction = (
                        (result["baseline_emissions"] - result["current_emissions"])
                        / result["baseline_emissions"]
                        * 100
                    )
                print(f"  Success: {result['success']}")
                print(f"  Tests passed: {result['test_passed']}")
                print(f"  Attempts: {result['attempts']}")
                print(f"  Emission reduction: {reduction:.1f}%")
                print(f"  Inference latency: {result['inference_duration']:.1f}s")
                print(f"  Tokens used: {result['inference_tokens']}")
            except Exception as e:
                print(f"  ERROR: {e}")
                func_results[engine] = {
                    "engine": engine,
                    "function_name": spec.function_name,
                    "success": False,
                    "test_passed": False,
                    "attempts": 0,
                    "baseline_emissions": 0.0,
                    "current_emissions": 0.0,
                    "inference_duration": 0.0,
                    "inference_tokens": 0,
                    "error": str(e),
                }

        all_results.append(func_results)

    return all_results


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(results: list[dict]) -> str:
    """Generate a markdown comparison report from benchmark results."""
    lines = []
    lines.append("# Engine Comparison Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Aggregate stats
    stats = {}
    for engine in ["claude", "crusoe"]:
        engine_results = [r[engine] for r in results if engine in r]
        successful = [r for r in engine_results if r.get("success")]
        tests_first_try = [r for r in engine_results if r.get("test_passed") and r.get("attempts", 0) == 1]

        reductions = []
        for r in engine_results:
            baseline = r.get("baseline_emissions", 0)
            current = r.get("current_emissions", 0)
            if baseline > 0 and r.get("success"):
                reductions.append((baseline - current) / baseline * 100)

        stats[engine] = {
            "optimized": len(successful),
            "total": len(engine_results),
            "avg_reduction": sum(reductions) / len(reductions) if reductions else 0,
            "avg_latency": (
                sum(r.get("inference_duration", 0) for r in engine_results)
                / len(engine_results)
                if engine_results
                else 0
            ),
            "avg_tokens": (
                sum(r.get("inference_tokens", 0) for r in engine_results)
                / len(engine_results)
                if engine_results
                else 0
            ),
            "first_try": len(tests_first_try),
            "total_attempts": sum(r.get("attempts", 0) for r in engine_results),
        }

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Metric | Claude (opus-4-6) | Crusoe (Qwen3-235B) |")
    lines.append("|---|---|---|")

    c, cr = stats["claude"], stats["crusoe"]
    lines.append(f"| Functions optimized | {c['optimized']}/{c['total']} | {cr['optimized']}/{cr['total']} |")
    lines.append(f"| Avg emission reduction | {c['avg_reduction']:.1f}% | {cr['avg_reduction']:.1f}% |")
    lines.append(f"| Avg inference latency | {c['avg_latency']:.1f}s | {cr['avg_latency']:.1f}s |")
    lines.append(f"| Avg tokens used | {c['avg_tokens']:.0f} | {cr['avg_tokens']:.0f} |")
    lines.append(f"| Tests passed on 1st try | {c['first_try']}/{c['total']} | {cr['first_try']}/{cr['total']} |")
    lines.append(f"| Total attempts | {c['total_attempts']} | {cr['total_attempts']} |")

    # Per-function results
    lines.append("\n## Per-Function Results\n")

    for r in results:
        fname = r["function_name"]
        lines.append(f"### `{fname}`\n")
        lines.append("| Metric | Claude | Crusoe |")
        lines.append("|---|---|---|")

        for metric, label in [
            ("success", "Optimized successfully"),
            ("test_passed", "Tests passed"),
            ("attempts", "Attempts"),
            ("baseline_emissions", "Baseline emissions (kg CO2eq)"),
            ("current_emissions", "Optimized emissions (kg CO2eq)"),
            ("inference_duration", "Inference latency (s)"),
            ("inference_tokens", "Tokens used"),
        ]:
            c_val = r.get("claude", {}).get(metric, "N/A")
            cr_val = r.get("crusoe", {}).get(metric, "N/A")

            if isinstance(c_val, float) and metric.endswith("emissions"):
                c_val = f"{c_val:.2e}"
                cr_val = f"{cr_val:.2e}" if isinstance(cr_val, float) else cr_val
            elif isinstance(c_val, float):
                c_val = f"{c_val:.1f}"
                cr_val = f"{cr_val:.1f}" if isinstance(cr_val, float) else cr_val

            lines.append(f"| {label} | {c_val} | {cr_val} |")

        # Emission reduction row
        for engine in ["claude", "crusoe"]:
            er = r.get(engine, {})
            baseline = er.get("baseline_emissions", 0)
            current = er.get("current_emissions", 0)
            if baseline > 0 and er.get("success"):
                reduction = (baseline - current) / baseline * 100
                if engine == "claude":
                    c_red = f"{reduction:.1f}%"
                else:
                    cr_red = f"{reduction:.1f}%"
            else:
                if engine == "claude":
                    c_red = "N/A"
                else:
                    cr_red = "N/A"
        lines.append(f"| Emission reduction | {c_red} | {cr_red} |")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting dual-engine benchmark...\n")
    start = time.time()

    results = run_benchmark()

    report = generate_report(results)
    duration = time.time() - start

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Benchmark complete in {duration:.0f}s")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
