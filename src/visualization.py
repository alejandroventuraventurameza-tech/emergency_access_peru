# -*- coding: utf-8 -*-
"""
visualization.py - Emergency Healthcare Access Inequality in Peru

Produces all static charts (matplotlib / seaborn) and static maps
(geopandas) to answer the four required analytical questions.

Chart selection rationale (documented per figure):
  Each chart was chosen because it answers a specific analytical
  question better than alternatives. See inline comments.
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "..", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
PALETTE = {
    "Underserved":  "#d73027",
    "Average":      "#fee090",
    "Well-served":  "#1a9850",
}
CLASS_ORDER = ["Underserved", "Average", "Well-served"]

sns.set_theme(style="whitegrid", font_scale=1.05)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_scores():
    return gpd.read_parquet(os.path.join(PROCESSED_DIR, "district_scores.parquet"))


# ── Q1: Territorial Availability ──────────────────────────────────────────────

def fig_ipress_distribution(df):
    """
    Chart: Distribution of IPRESS count per district (histogram + KDE).

    Why chosen: A histogram with KDE reveals the shape of the facility
    distribution across all 1,873 districts — in particular the strong
    right skew that characterizes health inequality (most districts have
    very few facilities; a handful of urban districts dominate).
    A box plot would hide this skew. A bar chart of all districts would
    be unreadable. This chart directly answers Q1 at a national level.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["n_ipress"], bins=40, kde=True, color="#2171b5", ax=ax)
    ax.axvline(df["n_ipress"].median(), color="red", ls="--", lw=1.5,
               label="Median = {:.0f}".format(df["n_ipress"].median()))
    ax.axvline(df["n_ipress"].mean(), color="orange", ls="--", lw=1.5,
               label="Mean = {:.1f}".format(df["n_ipress"].mean()))
    ax.set(title="Distribution of IPRESS Facilities per District",
           xlabel="Number of IPRESS Facilities", ylabel="Number of Districts")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_ipress_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG1] Saved:", os.path.basename(path))


def fig_top_bottom_facilities(df, n=20):
    """
    Chart: Top-N and Bottom-N districts by facility score (horizontal bars).

    Why chosen: Horizontal bar charts allow easy label reading for district
    names. Showing both extremes in one figure makes the inequality stark.
    A map alone would not communicate exact rankings. This directly names
    the districts that Q1 asks about.
    """
    top = df.nsmallest(n, "rank_baseline")[["distrito", "departamen", "n_ipress"]].copy()
    bot = df.nlargest(n, "rank_baseline")[["distrito", "departamen", "n_ipress"]].copy()
    top["label"] = top["distrito"] + " (" + top["departamen"] + ")"
    bot["label"] = bot["distrito"] + " (" + bot["departamen"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(top["label"][::-1], top["n_ipress"][::-1], color="#1a9850")
    axes[0].set(title="Top {} Best-Served Districts".format(n),
                xlabel="Number of IPRESS")
    axes[1].barh(bot["label"][::-1], bot["n_ipress"][::-1], color="#d73027")
    axes[1].set(title="Top {} Most Underserved Districts".format(n),
                xlabel="Number of IPRESS")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Territorial Availability: Facility Count Extremes", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_top_bottom_facilities.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG2] Saved:", os.path.basename(path))


def fig_dept_boxplot(df):
    """
    Chart: Box plot of n_ipress by department.

    Why chosen: Reveals within-department variation and inter-department
    differences simultaneously. A bar chart of means would obscure the
    spread. This answers Q1 at the department level and reveals which
    departments are internally unequal vs uniformly served.
    """
    dept_order = (df.groupby("departamen")["n_ipress"]
                  .median().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df, x="departamen", y="n_ipress",
                order=dept_order, palette="Blues", ax=ax, fliersize=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set(title="IPRESS Facility Count by Department (median-sorted)",
           xlabel="Department", ylabel="IPRESS per District")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_dept_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG3] Saved:", os.path.basename(path))


# ── Q2: Settlement Access ──────────────────────────────────────────────────────

def fig_distance_distribution(df):
    """
    Chart: Distribution of mean CCPP-to-nearest-IPRESS distance (histogram).

    Why chosen: Distance is the core spatial access proxy. Its distribution
    shows how many districts face long travel to any facility. A CDF overlay
    makes thresholds (5 km, 10 km) directly readable, which a box plot would
    not. Directly answers Q2 at national level.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = df["mean_dist_km"].dropna()
    sns.histplot(vals, bins=50, kde=True, color="#6baed6", ax=ax)
    ax.axvline(5,  color="red",    ls="--", lw=1.5, label="5 km threshold")
    ax.axvline(10, color="orange", ls="--", lw=1.5, label="10 km threshold")
    ax.axvline(vals.median(), color="black", ls=":", lw=1.5,
               label="Median = {:.1f} km".format(vals.median()))
    ax.set(title="Distribution of Mean Distance from Populated Centers to Nearest IPRESS",
           xlabel="Mean Distance (km)", ylabel="Number of Districts")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_distance_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG4] Saved:", os.path.basename(path))


def fig_access_scatter(df):
    """
    Chart: Scatter — n_ccpp vs mean_dist_km, colored by classification.

    Why chosen: Reveals the relationship between district size (proxied
    by n_ccpp) and access quality. Large districts (many CCPP) that also
    have high mean distance are the most at-risk for equity gaps. A heatmap
    or line chart could not show individual district positions. Directly
    answers Q2 at the district level.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in CLASS_ORDER:
        sub = df[df["class_baseline"] == cls]
        ax.scatter(sub["n_ccpp"], sub["mean_dist_km"],
                   c=PALETTE[cls], label=cls, alpha=0.6, s=15)
    ax.set(title="Settlement Size vs. Access Distance by District",
           xlabel="Number of Populated Centers (CCPP)",
           ylabel="Mean Distance to Nearest IPRESS (km)")
    ax.legend(title="Classification")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_access_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG5] Saved:", os.path.basename(path))


# ── Q3: District Comparison ───────────────────────────────────────────────────

def fig_score_ranking(df, n=25):
    """
    Chart: Horizontal bar — top and bottom N districts by baseline score.

    Why chosen: The composite score combines all three components into one
    comparable value. Horizontal bars with color coding by classification
    tier make the ranking immediately interpretable. A table alone would
    not convey the magnitude differences. Directly answers Q3.
    """
    top = df.nsmallest(n, "rank_baseline").copy()
    bot = df.nlargest(n, "rank_baseline").copy()

    for sub in [top, bot]:
        sub["label"] = sub["distrito"] + "\n(" + sub["departamen"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    colors_top = [PALETTE[c] for c in top["class_baseline"]]
    colors_bot = [PALETTE[c] for c in bot["class_baseline"]]

    axes[0].barh(top["label"][::-1], top["score_baseline"][::-1], color=colors_top[::-1])
    axes[0].set(title="Top {} Best-Served (Baseline Score)".format(n),
                xlabel="Access Score")

    axes[1].barh(bot["label"][::-1], bot["score_baseline"][::-1], color=colors_bot[::-1])
    axes[1].set(title="Top {} Most Underserved (Baseline Score)".format(n),
                xlabel="Access Score")

    patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3, title="Classification")
    fig.suptitle("District Comparison: Emergency Healthcare Access Score", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(FIGURES_DIR, "fig6_score_ranking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG6] Saved:", os.path.basename(path))


def fig_component_heatmap(df, n=40):
    """
    Chart: Heatmap of the 3 normalized components for top/bottom districts.

    Why chosen: A heatmap exposes which component drives each district's
    classification — i.e., whether a district is underserved because of
    few facilities, low activity, or poor spatial access, or all three.
    This nuance is invisible in a single composite bar. Essential for Q3.
    """
    top = df.nsmallest(n // 2, "rank_baseline")
    bot = df.nlargest(n // 2, "rank_baseline")
    combined = pd.concat([top, bot])
    combined["label"] = combined["distrito"] + " (" + combined["departamen"] + ")"
    combined = combined.set_index("label")

    heat_data = combined[["b_facility", "b_activity", "b_access"]].rename(columns={
        "b_facility": "Facility\nAvailability",
        "b_activity": "Emergency\nActivity",
        "b_access":   "Spatial\nAccess",
    })

    fig, ax = plt.subplots(figsize=(9, 14))
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.4, ax=ax)
    ax.set(title="Component Scores — Top & Bottom {} Districts".format(n // 2))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_component_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG7] Saved:", os.path.basename(path))


# ── Q4: Sensitivity Analysis ──────────────────────────────────────────────────

def fig_spec_scatter(df):
    """
    Chart: Scatter — baseline score vs alternative score, per district.

    Why chosen: Directly visualizes agreement and disagreement between both
    specifications. Points far from the diagonal are districts whose
    classification is sensitive to the analytical choices made. A rank
    correlation table alone would not reveal which districts flip. This is
    the core Q4 visualization.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    flipped = df["class_baseline"] != df["class_alternative"]

    ax.scatter(df.loc[~flipped, "score_baseline"],
               df.loc[~flipped, "score_alternative"],
               c="#4575b4", alpha=0.4, s=12, label="Stable classification")
    ax.scatter(df.loc[flipped, "score_baseline"],
               df.loc[flipped, "score_alternative"],
               c="#d73027", alpha=0.7, s=20, label="Classification changed")

    lims = [0, max(df["score_baseline"].max(), df["score_alternative"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect agreement")
    ax.set(xlim=lims, ylim=lims,
           title="Baseline vs Alternative Score — Sensitivity Analysis",
           xlabel="Baseline Score", ylabel="Alternative Score")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig8_spec_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG8] Saved:", os.path.basename(path))


def fig_rank_change(df):
    """
    Chart: Histogram of rank change (alternative - baseline).

    Why chosen: A histogram of rank deltas quantifies how much the
    ranking changes when the specification changes. A symmetric narrow
    distribution indicates robustness; a wide or skewed one indicates
    sensitivity. This is the most compact summary of Q4.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    changes = df["rank_alternative"] - df["rank_baseline"]
    sns.histplot(changes, bins=50, kde=True, color="#8073ac", ax=ax)
    ax.axvline(0, color="black", ls="--", lw=1.5, label="No change")
    ax.set(title="Rank Change: Alternative vs Baseline Specification",
           xlabel="Rank Change (Alternative - Baseline)",
           ylabel="Number of Districts")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig9_rank_change.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG9] Saved:", os.path.basename(path))


# ── Static maps ───────────────────────────────────────────────────────────────

def map_baseline_score(df):
    """Static choropleth: baseline access score per district."""
    fig, ax = plt.subplots(figsize=(10, 14))
    df.plot(column="score_baseline", cmap="RdYlGn", linewidth=0.1,
            edgecolor="grey", legend=True, ax=ax,
            legend_kwds={"label": "Baseline Access Score", "shrink": 0.6})
    ax.set_axis_off()
    ax.set_title("Emergency Healthcare Access — Baseline Score\nby District (Peru)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "map1_baseline_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[MAP1] Saved:", os.path.basename(path))


def map_classification(df):
    """Static choropleth: 3-tier classification (baseline)."""
    color_map = {c: PALETTE[c] for c in CLASS_ORDER}
    df = df.copy()
    df["color"] = df["class_baseline"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 14))
    df.plot(color=df["color"], linewidth=0.1, edgecolor="grey", ax=ax)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc="lower left", title="Classification", fontsize=10)
    ax.set_axis_off()
    ax.set_title("Emergency Healthcare Access — District Classification\n(Baseline Specification)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "map2_classification.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[MAP2] Saved:", os.path.basename(path))


def map_mean_distance(df):
    """Static choropleth: mean distance to nearest IPRESS."""
    fig, ax = plt.subplots(figsize=(10, 14))
    df.plot(column="mean_dist_km", cmap="OrRd", linewidth=0.1,
            edgecolor="grey", legend=True, ax=ax,
            legend_kwds={"label": "Mean Distance to IPRESS (km)", "shrink": 0.6},
            missing_kwds={"color": "lightgrey", "label": "No data"})
    ax.set_axis_off()
    ax.set_title("Mean Distance from Populated Centers\nto Nearest IPRESS by District", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "map3_mean_distance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[MAP3] Saved:", os.path.basename(path))


def map_spec_comparison(df):
    """Static map: districts that change classification between specs."""
    df = df.copy()
    df["stability"] = df.apply(
        lambda r: "Stable" if r["class_baseline"] == r["class_alternative"]
        else "Changed", axis=1)
    color_map = {"Stable": "#4575b4", "Changed": "#d73027"}
    df["color"] = df["stability"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 14))
    df.plot(color=df["color"], linewidth=0.1, edgecolor="grey", ax=ax)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc="lower left", title="Classification", fontsize=10)
    ax.set_axis_off()
    ax.set_title("Districts Where Classification Changes\nBetween Baseline and Alternative Spec", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "map4_spec_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[MAP4] Saved:", os.path.basename(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all():
    """Generate all static charts and maps."""
    print("\n--- Loading district scores ---")
    df = load_scores()

    print("\n--- Q1: Territorial Availability ---")
    fig_ipress_distribution(df)
    fig_top_bottom_facilities(df)
    fig_dept_boxplot(df)

    print("\n--- Q2: Settlement Access ---")
    fig_distance_distribution(df)
    fig_access_scatter(df)

    print("\n--- Q3: District Comparison ---")
    fig_score_ranking(df)
    fig_component_heatmap(df)

    print("\n--- Q4: Sensitivity Analysis ---")
    fig_spec_scatter(df)
    fig_rank_change(df)

    print("\n--- Static Maps ---")
    map_baseline_score(df)
    map_classification(df)
    map_mean_distance(df)
    map_spec_comparison(df)

    print("\n[OK] All figures saved to output/figures/")


if __name__ == "__main__":
    generate_all()
