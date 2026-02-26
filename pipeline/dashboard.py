"""Streamlit interactive dashboard for carbon emissions ML pipeline."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=60)
def load_analysis() -> dict:
    path = DATA_DIR / "analysis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_emissions() -> pd.DataFrame:
    path = DATA_DIR / "emissions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["baseline_emissions_kg", "optimized_emissions_kg", "reduction_pct",
                "loc", "cyclomatic_complexity", "num_loops"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_overview(analysis: dict, df: pd.DataFrame) -> None:
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    n_measured = analysis.get("n_functions_measured", df["baseline_emissions_kg"].notna().sum()
                               if "baseline_emissions_kg" in df.columns else 0)
    n_optimized = analysis.get("n_functions_optimized", 0)

    avg_baseline = df["baseline_emissions_kg"].mean() if "baseline_emissions_kg" in df.columns else None
    avg_baseline_str = f"{avg_baseline:.2e} kg" if (avg_baseline is not None and pd.notna(avg_baseline) and avg_baseline > 0) else "N/A"

    top20 = df[df["optimizer_ran"].astype(str) == "True"]["reduction_pct"].dropna() \
        if "optimizer_ran" in df.columns else pd.Series(dtype=float)
    avg_reduction_str = f"{top20.mean():.1f}%" if not top20.empty else "N/A"

    col1.metric("Functions Scanned", n_measured)
    col2.metric("Functions Optimized", n_optimized)
    col3.metric("Avg Baseline Emissions", avg_baseline_str)
    col4.metric("Avg Reduction (Top-20)", avg_reduction_str)

    # Summary stats table
    if "stats_by_category" in analysis and analysis["stats_by_category"]:
        st.subheader("Emissions by Category")
        stats = analysis["stats_by_category"]
        rows = []
        for cat, s in stats.items():
            rows.append({
                "Category": cat,
                "Count": s.get("n", 0),
                "Mean (kg CO2eq)": f"{s.get('mean', 0):.2e}",
                "Median (kg CO2eq)": f"{s.get('median', 0):.2e}",
                "P90 (kg CO2eq)": f"{s.get('p90', 0):.2e}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_pattern_heatmap(analysis: dict) -> None:
    st.header("Pattern Heatmap")
    st.caption("Feature importances from Random Forest predicting log(baseline_emissions_kg). "
               "Higher = stronger predictor of total emissions.")

    importance = analysis.get("feature_importance_baseline", {})
    if not importance:
        st.warning("No feature importance data available. Run `--analyze` stage first.")
        return

    features = list(importance.keys())
    values = list(importance.values())

    fig = px.bar(
        x=values,
        y=features,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title="Which Code Patterns Drive High Emissions?",
        color=values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False,
                      coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Also show per-loc importance
    per_loc = analysis.get("feature_importance_per_loc", {})
    if per_loc:
        st.subheader("Density-Adjusted Importance (per line of code)")
        st.caption("Reveals patterns that are carbon-dense independent of function length.")
        feat2 = list(per_loc.keys())
        vals2 = list(per_loc.values())
        fig2 = px.bar(
            x=vals2,
            y=feat2,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title="What Makes Code Carbon-Dense? (emissions / LOC)",
            color=vals2,
            color_continuous_scale="Oranges",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False,
                           coloraxis_showscale=False, height=500)
        st.plotly_chart(fig2, use_container_width=True)


def render_category_violin(df: pd.DataFrame) -> None:
    st.header("ML vs Data Processing")
    st.caption("Distribution of baseline emissions per function category (log scale).")

    if df.empty or "baseline_emissions_kg" not in df.columns:
        st.warning("No emissions data available.")
        return

    plot_df = df[df["baseline_emissions_kg"].notna() & (df["baseline_emissions_kg"] > 0)].copy()
    if plot_df.empty:
        st.warning("No valid baseline measurements yet.")
        return

    fig = px.violin(
        plot_df,
        x="category",
        y="baseline_emissions_kg",
        box=True,
        points="all",
        log_y=True,
        color="category",
        labels={"baseline_emissions_kg": "Baseline Emissions (kg CO2eq)", "category": "Category"},
        title="Emissions Distribution by Category (log scale)",
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_savings_histogram(df: pd.DataFrame) -> None:
    st.header("Savings Distribution")
    st.caption("Emissions reduction achieved by the Claude optimizer on the top-20 functions.")

    if df.empty or "optimizer_ran" not in df.columns:
        st.warning("No optimizer data available.")
        return

    top20 = df[df["optimizer_ran"].astype(str) == "True"].dropna(subset=["reduction_pct"])
    if top20.empty:
        st.warning("No optimizer results yet. Run `--measure` stage.")
        return

    fig = px.histogram(
        top20,
        x="reduction_pct",
        nbins=10,
        labels={"reduction_pct": "Emissions Reduction (%)", "count": "Count"},
        title="Emissions Reduction Distribution (Top-20 Optimizer Runs)",
        color_discrete_sequence=["#2ecc71"],
    )
    fig.add_vline(x=top20["reduction_pct"].mean(), line_dash="dash",
                  annotation_text=f"Mean: {top20['reduction_pct'].mean():.1f}%")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Top-20 Results Summary")
    summary_cols = ["function_name", "category", "baseline_emissions_kg",
                    "optimized_emissions_kg", "reduction_pct"]
    available = [c for c in summary_cols if c in top20.columns]
    st.dataframe(
        top20[available].sort_values("reduction_pct", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )


def render_top_wins(analysis: dict) -> None:
    st.header("Top Wins")
    st.caption("Functions with highest emissions reduction, showing before/after optimized code.")

    top20 = analysis.get("top20_results", [])
    if not top20:
        st.warning("No top-20 results in analysis.json. Run `--analyze` stage.")
        return

    sorted_results = sorted(top20, key=lambda r: r.get("reduction_pct", 0), reverse=True)

    for i, result in enumerate(sorted_results):
        reduction = result.get("reduction_pct", 0)
        name = result.get("function_name", "unknown")
        source_file = result.get("source_file", "")
        baseline = result.get("baseline_emissions_kg")
        optimized = result.get("optimized_emissions_kg")

        label = f"{i + 1}. `{name}` — {reduction:.1f}% reduction"
        if baseline and optimized:
            label += f" ({baseline:.2e} → {optimized:.2e} kg CO2eq)"

        with st.expander(label):
            if source_file:
                st.caption(f"Source: {source_file}")

            original = result.get("original_source", "")
            optimized_src = result.get("optimized_source", "")

            if original or optimized_src:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.code(original or "(not available)", language="python")
                with col2:
                    st.subheader("Optimized")
                    st.code(optimized_src or "(not available)", language="python")
            else:
                st.info("Source code not stored in analysis.json.")


def render_cluster_explorer(analysis: dict, df: pd.DataFrame) -> None:
    st.header("Cluster Explorer")
    st.caption("Functions grouped by code pattern similarity. "
               "Axes show size (LOC) vs complexity (cyclomatic).")

    if df.empty or "cluster" not in df.columns:
        st.warning("No cluster data available. Run `--analyze` stage.")
        return

    plot_df = df[df["cluster"].notna() & df["loc"].notna() & df["cyclomatic_complexity"].notna()].copy()
    plot_df["cluster"] = plot_df["cluster"].astype(int).astype(str)

    if plot_df.empty:
        st.warning("No clustered data to display.")
        return

    hover_cols = [c for c in ["function_name", "baseline_emissions_kg", "category", "source_file"]
                  if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x="loc",
        y="cyclomatic_complexity",
        color="cluster",
        hover_data=hover_cols,
        labels={"loc": "Lines of Code", "cyclomatic_complexity": "Cyclomatic Complexity",
                "cluster": "Cluster"},
        title="Function Clusters: Size vs Complexity",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Cluster profiles table
    profiles = analysis.get("cluster_profiles", [])
    if profiles:
        st.subheader("Cluster Profiles")
        profile_rows = []
        for p in profiles:
            profile_rows.append({
                "Cluster": p["cluster_id"],
                "Members": p["n_members"],
                "Mean Emissions (kg CO2eq)": f"{p['mean_emissions']:.2e}" if p.get("mean_emissions") else "N/A",
                "Mean LOC": f"{p['mean_loc']:.0f}" if p.get("mean_loc") else "N/A",
            })
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Carbon Emissions ML Pipeline",
        page_icon="🌿",
        layout="wide",
    )
    st.title("Carbon Emissions Analysis Dashboard")
    st.caption("Analyzes which Python code patterns are most carbon-costly and how much the optimizer reduces them.")

    analysis = load_analysis()
    df = load_emissions()

    if not analysis and df.empty:
        st.error("No data found. Run the pipeline first:\n"
                 "```\npython pipeline/run_pipeline.py --all\n```")
        return

    tab_labels = [
        "Overview",
        "Pattern Heatmap",
        "ML vs Data Processing",
        "Savings Distribution",
        "Top Wins",
        "Cluster Explorer",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview(analysis, df)
    with tabs[1]:
        render_pattern_heatmap(analysis)
    with tabs[2]:
        render_category_violin(df)
    with tabs[3]:
        render_savings_histogram(df)
    with tabs[4]:
        render_top_wins(analysis)
    with tabs[5]:
        render_cluster_explorer(analysis, df)


if __name__ == "__main__":
    main()
