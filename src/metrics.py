# -*- coding: utf-8 -*-
"""
metrics.py - Emergency Healthcare Access Inequality in Peru

Constructs district-level emergency healthcare access scores.

Two specifications are built and compared:

  BASELINE (spec='baseline'):
    - Facility component : IPRESS count per populated center (n_ipress / n_ccpp)
    - Activity component : total emergency attendances per IPRESS (2024 data)
    - Access component   : % of CCPP within 5 km of any IPRESS
    - Aggregation        : equal-weight average of 3 min-max normalized components
    - Classification     : quintile-based (Q1=underserved, Q5=well-served)

  ALTERNATIVE (spec='alternative'):
    - Facility component : IPRESS density per km2 (n_ipress / area_km2)
    - Activity component : total patients seen per CCPP (atendidos 2025 / n_ccpp)
    - Access component   : % of CCPP within 10 km of any IPRESS (more lenient)
    - Aggregation        : weighted (facility 25%, activity 25%, access 50%)
    - Classification     : quintile-based

Methodological justification:
  The baseline treats all three components equally and uses a stricter
  access threshold (5 km), which privileges districts with nearby care.
  The alternative down-weights facility presence and up-weights broad
  spatial coverage, penalizing districts where large populations remain
  far from any facility even beyond the 5 km mark. Comparing both
  surfaces districts where the choice of threshold or weighting matters.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_TABLES = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def minmax(series):
    """Min-max normalize a pandas Series to [0, 1]. NaN preserved as 0."""
    s = series.copy().fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mn) / (mx - mn)


def quintile_label(score_series):
    """Assign quintile labels Q1 (worst) to Q5 (best)."""
    labels = ["Q1-Underserved", "Q2-Below avg", "Q3-Average",
              "Q4-Above avg", "Q5-Well-served"]
    return pd.qcut(score_series, q=5, labels=labels, duplicates="drop")


def classify_3(score_series, low_pct=0.30, high_pct=0.70):
    """Classify districts into 3 tiers based on percentile thresholds."""
    low  = score_series.quantile(low_pct)
    high = score_series.quantile(high_pct)
    def _label(v):
        if v <= low:
            return "Underserved"
        elif v >= high:
            return "Well-served"
        else:
            return "Average"
    return score_series.map(_label)


# ── Data assembly ─────────────────────────────────────────────────────────────

def load_inputs():
    """
    Load all processed inputs needed for metric construction.

    Returns dict with:
        'spatial'  : district_spatial GeoDataFrame
        'em2024'   : emergencia 2024 district-level aggregation
        'em2025'   : emergencia 2025 district-level aggregation
    """
    spatial = gpd.read_parquet(os.path.join(PROCESSED_DIR, "district_spatial.parquet"))
    em2024  = pd.read_csv(os.path.join(PROCESSED_DIR, "emergencia_2024_district.csv"),
                          dtype={"ubigeo": str})
    em2025  = pd.read_csv(os.path.join(PROCESSED_DIR, "emergencia_2025_district.csv"),
                          dtype={"ubigeo": str})
    return {"spatial": spatial, "em2024": em2024, "em2025": em2025}


def assemble_base_table(inputs):
    """
    Merge spatial summary with emergency activity data and compute
    derived columns needed by both specifications.

    Returns GeoDataFrame with all raw components for both specs.
    """
    spatial = inputs["spatial"].copy()
    em2024  = inputs["em2024"][["ubigeo", "total_atenciones", "total_atendidos"]].copy()
    em2025  = inputs["em2025"][["ubigeo", "total_atenciones", "total_atendidos"]].copy()

    em2024 = em2024.rename(columns={
        "total_atenciones": "atenciones_2024",
        "total_atendidos":  "atendidos_2024",
    })
    em2025 = em2025.rename(columns={
        "total_atenciones": "atenciones_2025",
        "total_atendidos":  "atendidos_2025",
    })

    df = spatial.merge(em2024, on="ubigeo", how="left")
    df = df.merge(em2025, on="ubigeo", how="left")

    # Compute area in km2 from projected geometry
    df_proj = df.to_crs("EPSG:32718")
    df["area_km2"] = df_proj.geometry.area / 1e6

    # Derived raw components
    # Baseline
    df["fac_per_ccpp"]     = df["n_ipress"] / df["n_ccpp"].replace(0, np.nan)
    df["atenc_per_ipress"] = df["atenciones_2024"] / df["n_ipress"].replace(0, np.nan)

    # Alternative
    df["fac_per_km2"]      = df["n_ipress"] / df["area_km2"].replace(0, np.nan)
    df["atend_per_ccpp"]   = df["atendidos_2025"] / df["n_ccpp"].replace(0, np.nan)

    return df


# ── Baseline specification ────────────────────────────────────────────────────

def compute_baseline(df):
    """
    Baseline specification:
        score = (facility_norm + activity_norm + access_norm) / 3

    Components:
        facility_norm : minmax(n_ipress / n_ccpp)
        activity_norm : minmax(atenciones_2024 / n_ipress)
        access_norm   : minmax(pct_ccpp_within5km)

    Returns df with baseline columns added.
    """
    df = df.copy()
    df["b_facility"] = minmax(df["fac_per_ccpp"])
    df["b_activity"] = minmax(df["atenc_per_ipress"])
    df["b_access"]   = minmax(df["pct_ccpp_within5km"])

    df["score_baseline"] = (
        df["b_facility"] + df["b_activity"] + df["b_access"]
    ) / 3.0

    df["rank_baseline"]   = df["score_baseline"].rank(ascending=False, method="min").astype(int)
    df["class_baseline"]  = classify_3(df["score_baseline"])
    df["q_baseline"]      = quintile_label(df["score_baseline"])

    n_under = (df["class_baseline"] == "Underserved").sum()
    n_well  = (df["class_baseline"] == "Well-served").sum()
    print("[BASELINE] Underserved: {:,}  |  Well-served: {:,}  |  Average: {:,}".format(
        n_under, n_well, len(df) - n_under - n_well))

    return df


# ── Alternative specification ─────────────────────────────────────────────────

def compute_alternative(df):
    """
    Alternative specification:
        score = 0.25*facility_norm + 0.25*activity_norm + 0.50*access_norm

    Components:
        facility_norm : minmax(n_ipress / area_km2)   [density by area]
        activity_norm : minmax(atendidos_2025 / n_ccpp) [patients per center]
        access_norm   : minmax(pct_ccpp_within10km)   [lenient threshold]

    Rationale: spatial coverage is weighted double because ensuring any
    facility exists within reasonable reach of all populated centers is
    a more fundamental equity criterion than facility density per se.

    Returns df with alternative columns added.
    """
    df = df.copy()
    df["a_facility"] = minmax(df["fac_per_km2"])
    df["a_activity"] = minmax(df["atend_per_ccpp"])
    df["a_access"]   = minmax(df["pct_ccpp_within10km"])

    df["score_alternative"] = (
        0.25 * df["a_facility"] +
        0.25 * df["a_activity"] +
        0.50 * df["a_access"]
    )

    df["rank_alternative"]  = df["score_alternative"].rank(ascending=False, method="min").astype(int)
    df["class_alternative"] = classify_3(df["score_alternative"])
    df["q_alternative"]     = quintile_label(df["score_alternative"])

    n_under = (df["class_alternative"] == "Underserved").sum()
    n_well  = (df["class_alternative"] == "Well-served").sum()
    print("[ALTERNATIVE] Underserved: {:,}  |  Well-served: {:,}  |  Average: {:,}".format(
        n_under, n_well, len(df) - n_under - n_well))

    return df


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_specifications(df):
    """
    Compare baseline vs alternative specifications.

    Computes:
        - Spearman rank correlation between both scores
        - Rank change per district (alternative rank - baseline rank)
        - Classification flip: districts that change tier between specs

    Returns comparison DataFrame (non-spatial).
    """
    comp = df[[
        "ubigeo", "distrito", "departamen",
        "score_baseline", "rank_baseline", "class_baseline",
        "score_alternative", "rank_alternative", "class_alternative",
    ]].copy()

    comp["rank_change"]  = comp["rank_alternative"] - comp["rank_baseline"]
    comp["class_flipped"] = comp["class_baseline"] != comp["class_alternative"]

    rho, pval = spearmanr(comp["score_baseline"], comp["score_alternative"])

    print("[COMPARISON] Spearman rho: {:.4f}  (p={:.4e})".format(rho, pval))
    print("[COMPARISON] Districts that change classification: {:,} / {:,}".format(
        comp["class_flipped"].sum(), len(comp)))
    print("[COMPARISON] Largest rank improvement (alt vs base):")
    print(comp.nsmallest(5, "rank_change")[["distrito", "departamen",
          "rank_baseline", "rank_alternative", "rank_change"]].to_string(index=False))
    print("[COMPARISON] Largest rank drop (alt vs base):")
    print(comp.nlargest(5, "rank_change")[["distrito", "departamen",
          "rank_baseline", "rank_alternative", "rank_change"]].to_string(index=False))

    comp.to_csv(os.path.join(OUTPUT_TABLES, "spec_comparison.csv"), index=False)
    return comp, rho


# ── Main orchestrator ─────────────────────────────────────────────────────────

def build_metrics():
    """
    Full metrics pipeline: load -> assemble -> baseline -> alternative -> compare.

    Returns dict with:
        'district_scores' : full GeoDataFrame with both specs
        'comparison'      : non-spatial comparison DataFrame
        'spearman_rho'    : rank correlation between specs
    """
    print("\n--- Loading inputs ---")
    inputs = load_inputs()

    print("\n--- Assembling base table ---")
    df = assemble_base_table(inputs)

    print("\n--- Computing baseline specification ---")
    df = compute_baseline(df)

    print("\n--- Computing alternative specification ---")
    df = compute_alternative(df)

    print("\n--- Comparing specifications ---")
    comparison, rho = compare_specifications(df)

    # Save district scores
    out = os.path.join(PROCESSED_DIR, "district_scores.parquet")
    df.to_parquet(out)
    print("\n[OK] Saved district_scores.parquet")

    # Save flat table for Streamlit
    cols_export = [
        "ubigeo", "distrito", "departamen", "provincia",
        "n_ipress", "n_ccpp", "area_km2",
        "mean_dist_km", "median_dist_km",
        "pct_ccpp_within5km", "pct_ccpp_within10km",
        "atenciones_2024", "atendidos_2025",
        "score_baseline", "rank_baseline", "class_baseline", "q_baseline",
        "score_alternative", "rank_alternative", "class_alternative", "q_alternative",
    ]
    df[[c for c in cols_export if c in df.columns]].to_csv(
        os.path.join(OUTPUT_TABLES, "district_scores.csv"), index=False
    )
    print("[OK] Saved output/tables/district_scores.csv")

    return {
        "district_scores": df,
        "comparison":      comparison,
        "spearman_rho":    rho,
    }


if __name__ == "__main__":
    results = build_metrics()
    df = results["district_scores"]
    print("\nTop 10 best-served districts (baseline):")
    top = df.nsmallest(10, "rank_baseline")[
        ["distrito", "departamen", "score_baseline", "rank_baseline", "class_baseline"]
    ]
    print(top.to_string(index=False))
    print("\nBottom 10 most underserved districts (baseline):")
    bot = df.nlargest(10, "rank_baseline")[
        ["distrito", "departamen", "score_baseline", "rank_baseline", "class_baseline"]
    ]
    print(bot.to_string(index=False))
