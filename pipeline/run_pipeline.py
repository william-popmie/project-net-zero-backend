"""CLI orchestrator for the carbon emissions ML pipeline."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `pipeline` is importable as a package
_REPO_ROOT_EARLY = Path(__file__).parent.parent
if str(_REPO_ROOT_EARLY) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_EARLY))

from dotenv import load_dotenv

# Load .env from repo root before anything else
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
FUNCTIONS_CSV = DATA_DIR / "functions.csv"
EMISSIONS_CSV = DATA_DIR / "emissions.csv"
ANALYSIS_JSON = DATA_DIR / "analysis.json"


def stage_scrape(token: str) -> None:
    from pipeline.scraper import scrape  # lazy import

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = scrape(token=token, output_dir=RAW_DIR)
    print(f"[scrape] Done. {len(files)} files in {RAW_DIR}")


def stage_extract() -> None:
    from pipeline.feature_extractor import run_extraction  # lazy import

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = run_extraction(raw_dir=RAW_DIR, output_csv=FUNCTIONS_CSV)
    print(f"[extract] Done. {len(df)} functions extracted -> {FUNCTIONS_CSV}")


def stage_baseline() -> None:
    """Baseline scan only — skips already-succeeded functions, retries failures."""
    from pipeline.emissions_runner import run_baseline_scan  # lazy import

    df = run_baseline_scan(functions_csv=FUNCTIONS_CSV, emissions_csv=EMISSIONS_CSV, raw_dir=RAW_DIR)
    n_success = df["baseline_emissions_kg"].notna().sum() if "baseline_emissions_kg" in df.columns else 0
    print(f"[baseline] Done: {n_success}/{len(df)} succeeded")


def stage_measure(use_hints: bool = False) -> None:
    from pipeline.emissions_runner import run_baseline_scan, run_top20_optimizer  # lazy import

    df = run_baseline_scan(functions_csv=FUNCTIONS_CSV, emissions_csv=EMISSIONS_CSV, raw_dir=RAW_DIR)
    n_success = df["baseline_emissions_kg"].notna().sum() if "baseline_emissions_kg" in df.columns else 0
    print(f"[measure] Baseline scan done: {n_success}/{len(df)} succeeded")

    df = run_top20_optimizer(emissions_csv=EMISSIONS_CSV, functions_csv=FUNCTIONS_CSV, use_hints=use_hints)
    n_optimized = (df["optimizer_ran"].astype(str) == "True").sum() if "optimizer_ran" in df.columns else 0
    print(f"[measure] Optimizer done: {n_optimized} functions optimized")


def stage_analyze() -> None:
    from pipeline.analysis import run_analysis  # lazy import

    run_analysis(emissions_csv=EMISSIONS_CSV, output_json=ANALYSIS_JSON)
    print(f"[analyze] Done -> {ANALYSIS_JSON}")


def stage_dashboard() -> None:
    dashboard_path = Path(__file__).parent / "dashboard.py"
    print(f"[dashboard] Launching Streamlit: streamlit run {dashboard_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        cwd=str(_REPO_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carbon Emissions ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages (run in order):
  --scrape     Fetch Python files from GitHub → pipeline/data/raw/
  --extract    Extract function features      → pipeline/data/functions.csv
  --measure    Measure baseline emissions +   → pipeline/data/emissions.csv
               run optimizer on top-20
  --analyze    ML analysis + clustering       → pipeline/data/analysis.json
  --dashboard  Launch Streamlit dashboard

Examples:
  python pipeline/run_pipeline.py --all --token ghp_xxx
  python pipeline/run_pipeline.py --scrape --token ghp_xxx
  python pipeline/run_pipeline.py --extract --measure --analyze
  python pipeline/run_pipeline.py --dashboard
""",
    )
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"),
                        help="GitHub personal access token (or set GITHUB_TOKEN in .env)")
    parser.add_argument("--scrape", action="store_true", help="Scrape Python files from GitHub")
    parser.add_argument("--extract", action="store_true", help="Extract function features")
    parser.add_argument("--baseline", action="store_true", help="Baseline scan only (retry failures, skip successes)")
    parser.add_argument("--measure", action="store_true", help="Run baseline + top-20 optimizer")
    parser.add_argument("--hints", action="store_true",
                        help="Inject AST-derived optimization hints into the Claude prompt (used with --measure or --all)")
    parser.add_argument("--analyze", action="store_true", help="Run ML analysis")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all stages in sequence")

    args = parser.parse_args()

    # If no stage specified, print help
    if not any([args.scrape, args.extract, args.baseline, args.measure,
                args.analyze, args.dashboard, args.run_all]):
        parser.print_help()
        sys.exit(0)

    run_all = args.run_all

    if run_all or args.scrape:
        if not args.token:
            print("[ERROR] GitHub token required for --scrape. "
                  "Pass --token or set GITHUB_TOKEN in .env")
            sys.exit(1)
        stage_scrape(args.token)

    if run_all or args.extract:
        stage_extract()

    if args.baseline:
        stage_baseline()

    if run_all or args.measure:
        stage_measure(use_hints=args.hints)

    if run_all or args.analyze:
        stage_analyze()

    if run_all or args.dashboard:
        stage_dashboard()


if __name__ == "__main__":
    main()
