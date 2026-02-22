"""sklearn ML analysis + clustering on emissions data."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "loc", "has_for_loop", "has_while_loop", "has_nested_loops",
    "num_loops", "has_list_comp", "has_dict_comp", "has_generator",
    "has_numpy", "has_pandas", "has_torch", "has_tensorflow", "has_sklearn",
    "has_string_ops", "num_function_calls", "num_arithmetic_ops", "cyclomatic_complexity",
]


def load_data(emissions_csv: Path) -> pd.DataFrame:
    """Load and preprocess emissions.csv. Returns cleaned DataFrame."""
    df = pd.read_csv(emissions_csv)

    # Coerce numeric columns
    for col in ["baseline_emissions_kg", "optimized_emissions_kg", "reduction_pct",
                "loc", "num_loops", "num_function_calls", "num_arithmetic_ops", "cyclomatic_complexity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce boolean feature columns
    bool_cols = [c for c in FEATURE_COLS if c not in
                 {"loc", "num_loops", "num_function_calls", "num_arithmetic_ops", "cyclomatic_complexity"}]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: True if str(x).lower() in ("true", "1") else False)
            df[col] = df[col].astype(int)

    return df


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build numeric feature matrix with category encoding + imputation."""
    cols_available = [c for c in FEATURE_COLS if c in df.columns]

    X = df[cols_available].copy()

    # Encode category as integer if present
    if "category" in df.columns:
        le = LabelEncoder()
        X["category_enc"] = le.fit_transform(df["category"].fillna("other").astype(str))
        feature_names = cols_available + ["category_enc"]
    else:
        feature_names = cols_available

    # Impute any remaining NaN values
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X[feature_names] if "category_enc" in X.columns else X[feature_names])
    return X_arr, feature_names


def train_baseline_model(df: pd.DataFrame) -> dict[str, float]:
    """
    Train RandomForest predicting log(baseline_emissions_kg).
    Returns feature importance dict sorted by importance descending.
    """
    valid = df[df["baseline_emissions_kg"].notna() & (df["baseline_emissions_kg"] > 0)].copy()
    if len(valid) < 10:
        return {}

    X, feature_names = _build_feature_matrix(valid)
    y = np.log(valid["baseline_emissions_kg"].values + 1e-15)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = dict(zip(feature_names, rf.feature_importances_))
    # Drop category_enc from output (internal encoding artifact)
    importance.pop("category_enc", None)
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train_per_loc_model(df: pd.DataFrame) -> dict[str, float]:
    """
    Train RandomForest predicting log(baseline_emissions_kg / loc).
    This reveals code patterns that are dense-expensive, independent of function length.
    """
    valid = df[
        df["baseline_emissions_kg"].notna()
        & (df["baseline_emissions_kg"] > 0)
        & df["loc"].notna()
        & (df["loc"] > 0)
    ].copy()
    if len(valid) < 10:
        return {}

    X, feature_names = _build_feature_matrix(valid)
    y = np.log(valid["baseline_emissions_kg"].values / valid["loc"].values + 1e-15)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = dict(zip(feature_names, rf.feature_importances_))
    importance.pop("category_enc", None)
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train_reduction_model(df: pd.DataFrame) -> dict[str, float]:
    """
    Train RandomForest predicting reduction_pct on top-20 optimizer rows.
    Returns feature importance dict.
    """
    optimized = df[
        (df["optimizer_ran"].astype(str) == "True")
        & df["reduction_pct"].notna()
    ].copy()

    if len(optimized) < 5:
        return {}

    X, feature_names = _build_feature_matrix(optimized)
    y = optimized["reduction_pct"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = dict(zip(feature_names, rf.feature_importances_))
    importance.pop("category_enc", None)
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def cluster_functions(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Cluster all functions with valid features using KMeans.
    Adds 'cluster' column to df. Returns updated DataFrame.
    """
    valid_mask = df["baseline_emissions_kg"].notna()
    valid = df[valid_mask].copy()

    if len(valid) < k:
        df["cluster"] = pd.NA
        return df

    X, _ = _build_feature_matrix(valid)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df = df.copy()
    df["cluster"] = pd.NA
    df.loc[valid_mask, "cluster"] = labels
    return df


def _cluster_profiles(df: pd.DataFrame, k: int = 4) -> list[dict]:
    """Compute per-cluster summary statistics."""
    profiles = []
    if "cluster" not in df.columns:
        return profiles

    for cluster_id in range(k):
        members = df[df["cluster"] == cluster_id]
        if members.empty:
            continue

        top_features = {}
        for col in FEATURE_COLS:
            if col in members.columns:
                top_features[col] = float(members[col].mean())

        profiles.append({
            "cluster_id": int(cluster_id),
            "n_members": int(len(members)),
            "mean_emissions": float(members["baseline_emissions_kg"].mean())
            if members["baseline_emissions_kg"].notna().any() else None,
            "mean_loc": float(members["loc"].mean()) if members["loc"].notna().any() else None,
            "top_features": top_features,
        })

    return profiles


def compute_category_stats(df: pd.DataFrame) -> dict:
    """Return per-category emissions statistics."""
    stats = {}
    valid = df[df["baseline_emissions_kg"].notna() & (df["baseline_emissions_kg"] > 0)]
    for cat, group in valid.groupby("category"):
        vals = group["baseline_emissions_kg"]
        stats[str(cat)] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p90": float(vals.quantile(0.9)),
            "n": int(len(vals)),
        }
    return stats


def _build_top20_results(df: pd.DataFrame) -> list[dict]:
    """Build top-20 optimizer results list for analysis.json."""
    optimized = df[
        (df["optimizer_ran"].astype(str) == "True")
        & df["reduction_pct"].notna()
    ].copy()

    optimized["reduction_pct"] = pd.to_numeric(optimized["reduction_pct"], errors="coerce")
    optimized = optimized.dropna(subset=["reduction_pct"])
    optimized = optimized.sort_values("reduction_pct", ascending=False)

    results = []
    for _, row in optimized.iterrows():
        results.append({
            "function_id": str(row.get("function_id", "")),
            "function_name": str(row.get("function_name", "")),
            "source_file": str(row.get("source_file", "")),
            "baseline_emissions_kg": float(row["baseline_emissions_kg"])
            if pd.notna(row.get("baseline_emissions_kg")) else None,
            "optimized_emissions_kg": float(row["optimized_emissions_kg"])
            if pd.notna(row.get("optimized_emissions_kg")) else None,
            "reduction_pct": float(row["reduction_pct"]),
            "original_source": str(row.get("source_code", "")),
            "optimized_source": str(row.get("optimized_source", "")),
        })

    return results


def run_analysis(emissions_csv: Path, output_json: Path) -> dict:
    """
    Main entry point. Load emissions data, run ML analysis, save analysis.json.
    Also writes cluster labels back to emissions.csv.
    Returns the analysis dict.
    """
    print(f"[analysis] Loading {emissions_csv}...")
    df = load_data(emissions_csv)

    print(f"[analysis] Training baseline model ({df['baseline_emissions_kg'].notna().sum()} valid rows)...")
    fi_baseline = train_baseline_model(df)

    print("[analysis] Training per-loc model...")
    fi_per_loc = train_per_loc_model(df)

    print("[analysis] Training reduction model...")
    fi_reduction = train_reduction_model(df)

    print("[analysis] Clustering functions...")
    df = cluster_functions(df, k=4)

    # Write cluster labels back to emissions.csv
    df.to_csv(emissions_csv, index=False)

    print("[analysis] Computing category stats...")
    cat_stats = compute_category_stats(df)

    profiles = _cluster_profiles(df)
    top20 = _build_top20_results(df)

    n_measured = int(df["baseline_emissions_kg"].notna().sum())
    n_optimized = int((df["optimizer_ran"].astype(str) == "True").sum())

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_functions_measured": n_measured,
        "n_functions_optimized": n_optimized,
        "feature_importance_baseline": fi_baseline,
        "feature_importance_per_loc": fi_per_loc,
        "feature_importance_reduction": fi_reduction,
        "cluster_profiles": profiles,
        "stats_by_category": cat_stats,
        "top20_results": top20,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"[analysis] Saved analysis to {output_json}")

    return analysis
