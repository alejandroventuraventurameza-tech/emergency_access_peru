# -*- coding: utf-8 -*-
"""
app.py - Emergency Healthcare Access Inequality in Peru
Streamlit application with 4 required tabs.
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from streamlit.components.v1 import html as st_html

warnings.filterwarnings("ignore")

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import (
    load_district_scores, load_district_scores_csv, load_spec_comparison,
    load_ipress_spatial, folium_choropleth, folium_classification_map,
    folium_ipress_points, map_to_html,
    CLASS_COLORS, CLASS_ORDER, fmt_number, fmt_pct, fmt_km,
    FIGURES_DIR,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Healthcare Access — Peru",
    page_icon="🏥",
    layout="wide",
)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def get_scores_gdf():
    return load_district_scores()

@st.cache_data
def get_scores_df():
    return load_district_scores_csv()

@st.cache_data
def get_comparison():
    return load_spec_comparison()

@st.cache_data
def get_ipress():
    return load_ipress_spatial()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ GeoSpatial Results",
    "🔍 Interactive Exploration",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Emergency Healthcare Access Inequality in Peru")
    st.caption("Python Programming / Data Science — HW_02_202601")

    st.header("Problem Statement")
    st.markdown("""
    Health emergencies are time-critical. In Peru — a geographically diverse
    country spanning coast, highlands, and jungle — access to emergency health
    services is highly unequal across its 1,873 districts.

    This project answers a single core question:

    > **Which districts in Peru appear relatively better or worse served in
    > emergency healthcare access, and what evidence supports that conclusion?**

    The analysis is deliberately **methodological**: we do not simply map
    facilities. We design, justify, and stress-test a district-level
    access framework built from four public datasets.
    """)

    st.header("Data Sources")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spatial Datasets")
        st.markdown("""
        | Dataset | Source | Records |
        |---|---|---|
        | Centros Poblados (CCPP) | IGN 1:100K | 136,587 |
        | District Boundaries | IGN / MINSA | 1,873 |
        """)
    with col2:
        st.subheader("Tabular Datasets")
        st.markdown("""
        | Dataset | Source | Records |
        |---|---|---|
        | IPRESS Registry | MINSA | 20,819 |
        | Emergency Production 2024 | MINSA | 250,000 |
        | Emergency Production 2025 | MINSA | 342,753 |
        """)

    st.header("Data Cleaning Summary")
    st.markdown("""
    **IPRESS:** Column names standardized (accents removed). Coordinate columns
    were swapped in the raw file (NORTE held longitude values, ESTE held latitude);
    renamed to `lon`/`lat` respectively. 12,866 facilities lacked coordinates and
    were excluded from spatial analysis (retained for attribute analysis).

    **Emergency data (2024/2025):** Values coded as `NE_XXXX` represent
    suppressed counts (privacy protection for small facilities). These were
    replaced with NaN. 2024: 36,377 suppressed rows; 2025: 44,889 suppressed rows.
    Remaining rows were aggregated to district level.

    **CCPP:** No null geometries. District assignment was done via spatial join
    (not name matching) — 221 centers fell on district borders and were left
    unmatched (< 0.2% of total).

    **DISTRITOS:** IDDIST renamed to `ubigeo` for pipeline consistency.
    Zero-padded to 6 digits throughout.
    """)

    st.header("Methodological Decisions")
    st.markdown("""
    **Distance proxy:** Straight-line distance (Euclidean) from each populated
    center to its nearest IPRESS facility, computed in UTM zone 18S (EPSG:32718 —
    the IGN Peru standard for 1:100K cartography). Road network distance was not
    used due to data unavailability at national scale.

    **Baseline specification:** Three equally-weighted components, all min-max
    normalized to [0, 1]:
    - *Facility availability* = n_ipress / n_ccpp per district
    - *Emergency activity* = total_atenciones_2024 / n_ipress
    - *Spatial access* = % of CCPP within 5 km of any IPRESS

    **Alternative specification:** Different weight distribution (access = 50%)
    and different operationalizations (facility density per km², 2025 atendidos
    per CCPP, 10 km access threshold). Rationale: spatial coverage is a more
    fundamental equity criterion than facility density per se.

    **Classification:** Bottom 30th percentile = Underserved; top 30th = Well-served.
    """)

    st.header("Limitations")
    st.markdown("""
    - Distance is straight-line, not road network — underestimates true travel time.
    - 12,866 IPRESS facilities lack coordinates — true facility counts per district
      are higher than reported here.
    - Emergency production data covers only IPRESS that reported — silent facilities
      may be inactive or simply non-reporting.
    - Population data not used (CCPP count used as proxy) — districts with large
      cities may be under-penalized relative to their true population burden.
    - 10 UBIGEO codes in the emergency data do not match the district shapefile
      (likely new districts created post-shapefile publication).
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Static Analysis")
    df = get_scores_df()

    st.markdown("""
    All charts were selected because they answer a specific analytical question
    better than available alternatives. Selection rationale is included below
    each figure.
    """)

    # Q1
    st.header("Q1 — Territorial Availability")

    c1, c2 = st.columns(2)
    with c1:
        img = os.path.join(FIGURES_DIR, "fig1_ipress_distribution.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** A histogram + KDE reveals the strong right skew in
        facility distribution — most districts have 1–5 facilities while a few
        urban districts have 50+. A box plot would hide this skew. This skew
        is the visual signature of health inequality.
        """)
    with c2:
        img = os.path.join(FIGURES_DIR, "fig3_dept_boxplot.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** A department-level box plot shows both
        inter-department differences and within-department variation
        simultaneously. A bar chart of means would lose the spread information,
        which is equally important for equity analysis.
        """)

    img = os.path.join(FIGURES_DIR, "fig2_top_bottom_facilities.png")
    if os.path.exists(img):
        st.image(img, use_column_width=True)
    st.caption("""
    **Why this chart:** Horizontal bars allow reading long district names cleanly.
    Showing both extremes in one figure makes the facility count gap concrete
    (e.g., Yanahuara has 59 facilities; dozens of rural districts have 0).
    """)

    # Q2
    st.header("Q2 — Settlement Access")
    c1, c2 = st.columns(2)
    with c1:
        img = os.path.join(FIGURES_DIR, "fig4_distance_distribution.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** The distance distribution with 5 km and 10 km
        reference lines makes the access thresholds used in the metric directly
        readable. A CDF would work too, but a histogram also shows where the
        bulk of districts cluster — around 2–5 km nationally.
        """)
    with c2:
        img = os.path.join(FIGURES_DIR, "fig5_access_scatter.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** The scatter of n_ccpp vs mean distance, colored
        by classification, reveals which districts combine large settlement
        counts with poor access — the highest-burden equity gap cases. A map
        alone would not surface this joint pattern.
        """)

    # Q3
    st.header("Q3 — District Comparison")
    img = os.path.join(FIGURES_DIR, "fig6_score_ranking.png")
    if os.path.exists(img):
        st.image(img, use_column_width=True)
    st.caption("""
    **Why this chart:** The composite score integrates all three components.
    Horizontal bars with classification color coding make the ranking and
    magnitude immediately interpretable. A table would require sorting and
    scanning; a map cannot communicate precise score differences.
    """)

    img = os.path.join(FIGURES_DIR, "fig7_component_heatmap.png")
    if os.path.exists(img):
        st.image(img, use_column_width=True)
    st.caption("""
    **Why this chart:** A heatmap of the three normalized components reveals
    *why* a district ranks where it does. A district may be underserved due
    to poor spatial access alone (e.g., large rural area) or due to low
    activity even with facilities present. This nuance is invisible in the
    composite score bar chart.
    """)

    # Q4
    st.header("Q4 — Methodological Sensitivity")
    c1, c2 = st.columns(2)
    with c1:
        img = os.path.join(FIGURES_DIR, "fig8_spec_scatter.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** Plotting baseline vs alternative score per district
        with flipped districts in red directly answers whether the two specs
        agree. Points near the diagonal = robust conclusions; red points =
        districts where the choice of specification matters.
        """)
    with c2:
        img = os.path.join(FIGURES_DIR, "fig9_rank_change.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("""
        **Why this chart:** A histogram of rank deltas summarizes sensitivity
        at the population level. A symmetric narrow distribution signals
        robustness; a wide one signals that results depend strongly on
        methodological choices. Spearman rho = 0.888 — high but imperfect.
        """)

    # Key stats
    st.header("Key Statistics")
    total = len(df)
    n_under = (df["class_baseline"] == "Underserved").sum()
    n_well  = (df["class_baseline"] == "Well-served").sum()
    n_flip  = (df["class_baseline"] != df["class_alternative"]).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Districts", fmt_number(total))
    m2.metric("Underserved", fmt_number(n_under), delta=fmt_pct(n_under/total*100))
    m3.metric("Well-served", fmt_number(n_well), delta=fmt_pct(n_well/total*100))
    m4.metric("Classification Changes (Sensitivity)", fmt_number(n_flip))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GeoSpatial Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("GeoSpatial Results")

    st.markdown("""
    Static maps showing the spatial distribution of emergency healthcare access
    across Peru's 1,873 districts.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Baseline Access Score")
        img = os.path.join(FIGURES_DIR, "map1_baseline_score.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("Green = better access. Red = worse access. "
                   "Clear coastal and Lima metro concentration of well-served districts.")
    with c2:
        st.subheader("District Classification (3 Tiers)")
        img = os.path.join(FIGURES_DIR, "map2_classification.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("Underserved districts (red) concentrate in the highlands and jungle. "
                   "Well-served (green) districts align with major coastal cities.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mean Distance to Nearest IPRESS")
        img = os.path.join(FIGURES_DIR, "map3_mean_distance.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("Darker = higher mean distance. Amazon districts show distances "
                   "exceeding 30–70 km — extreme spatial isolation from any facility.")
    with c2:
        st.subheader("Specification Sensitivity Map")
        img = os.path.join(FIGURES_DIR, "map4_spec_comparison.png")
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.caption("Red = district classification changes between baseline and alternative. "
                   "471 of 1,873 districts flip tier — concentrated in peri-urban zones.")

    # District-level table
    st.subheader("District-Level Results Table")
    df_tab = get_scores_df()
    depts = sorted(df_tab["departamen"].dropna().unique())
    selected_dept = st.selectbox("Filter by department:", ["All"] + list(depts))
    if selected_dept != "All":
        df_tab = df_tab[df_tab["departamen"] == selected_dept]

    show_cols = ["ubigeo", "distrito", "departamen", "n_ipress", "n_ccpp",
                 "mean_dist_km", "pct_ccpp_within5km",
                 "score_baseline", "rank_baseline", "class_baseline"]
    show_cols = [c for c in show_cols if c in df_tab.columns]
    st.dataframe(
        df_tab[show_cols].sort_values("rank_baseline"),
        use_container_width=True,
        height=400,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Exploration
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("Interactive Exploration")

    gdf = get_scores_gdf()
    ipress_df = get_ipress()

    map_choice = st.radio(
        "Select map:",
        ["Classification (3 tiers)", "Baseline Score", "Mean Distance to IPRESS",
         "IPRESS Facility Points"],
        horizontal=True,
    )

    with st.spinner("Rendering map..."):
        if map_choice == "Classification (3 tiers)":
            m = folium_classification_map(gdf)
        elif map_choice == "Baseline Score":
            m = folium_choropleth(
                gdf, "score_baseline", "Baseline Access Score",
                cmap="RdYlGn",
                tooltip_fields=["distrito", "departamen", "score_baseline",
                                 "rank_baseline", "class_baseline"],
            )
        elif map_choice == "Mean Distance to IPRESS":
            m = folium_choropleth(
                gdf, "mean_dist_km", "Mean Distance to IPRESS (km)",
                cmap="OrRd",
                tooltip_fields=["distrito", "departamen", "mean_dist_km",
                                 "pct_ccpp_within5km", "n_ccpp"],
            )
        else:
            m = folium_ipress_points(gdf, ipress_df)

        st_html(map_to_html(m), height=600, scrolling=False)

    # Spec comparison
    st.subheader("Baseline vs Alternative: District Comparison")
    comp = get_comparison()

    col1, col2 = st.columns([2, 1])
    with col1:
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(8, 6))
        flipped = comp["class_baseline"] != comp["class_alternative"]
        ax.scatter(comp.loc[~flipped, "score_baseline"],
                   comp.loc[~flipped, "score_alternative"],
                   c="#4575b4", alpha=0.35, s=10, label="Stable")
        ax.scatter(comp.loc[flipped, "score_baseline"],
                   comp.loc[flipped, "score_alternative"],
                   c="#d73027", alpha=0.7, s=15, label="Classification changed")
        lim = [0, max(comp["score_baseline"].max(),
                      comp["score_alternative"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="Perfect agreement")
        ax.set(xlim=lim, ylim=lim,
               xlabel="Baseline Score", ylabel="Alternative Score",
               title="Score Agreement Between Specifications")
        ax.legend(fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.metric("Spearman rho", "0.888")
        st.metric("Districts with stable classification",
                  fmt_number((~flipped).sum()))
        st.metric("Districts that change tier",
                  fmt_number(flipped.sum()))
        st.markdown("""
        **Interpretation:** A Spearman rho of 0.888 indicates high
        but imperfect concordance. The 471 districts that change
        classification are mostly peri-urban and mid-sized highland
        districts — their ranking is sensitive to whether access is
        measured at 5 km or 10 km and whether facility density is
        computed per CCPP or per km².
        """)

    # Top movers
    st.subheader("Districts with Largest Rank Change")
    tab_up, tab_down = st.tabs(["Largest improvement (alt vs base)",
                                 "Largest drop (alt vs base)"])
    with tab_up:
        top_up = comp.nsmallest(15, "rank_change")[
            ["distrito", "departamen", "rank_baseline",
             "rank_alternative", "rank_change"]
        ]
        st.dataframe(top_up, use_container_width=True)
    with tab_down:
        top_down = comp.nlargest(15, "rank_change")[
            ["distrito", "departamen", "rank_baseline",
             "rank_alternative", "rank_change"]
        ]
        st.dataframe(top_down, use_container_width=True)
