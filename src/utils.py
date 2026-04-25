# -*- coding: utf-8 -*-
"""
utils.py - Emergency Healthcare Access Inequality in Peru

Shared helper functions used across the pipeline and Streamlit app.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "output", "figures")
TABLES_DIR    = os.path.join(BASE_DIR, "output", "tables")

CLASS_COLORS = {
    "Underserved":  "#d73027",
    "Average":      "#fee090",
    "Well-served":  "#1a9850",
}
CLASS_ORDER = ["Underserved", "Average", "Well-served"]


# ── Cached loaders (used by Streamlit with @st.cache_data) ───────────────────

def load_district_scores():
    """Load district scores GeoDataFrame from processed parquet."""
    return gpd.read_parquet(os.path.join(PROCESSED_DIR, "district_scores.parquet"))


def load_district_scores_csv():
    """Load district scores as plain DataFrame (faster for tables)."""
    return pd.read_csv(
        os.path.join(TABLES_DIR, "district_scores.csv"),
        dtype={"ubigeo": str},
    )


def load_spec_comparison():
    """Load baseline vs alternative comparison table."""
    return pd.read_csv(
        os.path.join(TABLES_DIR, "spec_comparison.csv"),
        dtype={"ubigeo": str},
    )


def load_ipress_spatial():
    """Load IPRESS facilities with valid coordinates."""
    return pd.read_csv(
        os.path.join(PROCESSED_DIR, "ipress_spatial.csv"),
        dtype={"ubigeo": str},
    )


# ── Folium map builders ───────────────────────────────────────────────────────

def folium_choropleth(gdf, column, title, cmap="RdYlGn", tooltip_fields=None):
    """
    Build a Folium choropleth map for a numeric column.

    Parameters:
        gdf           : GeoDataFrame with geometry and data columns
        column        : column name to choropleth
        title         : map title (shown as layer control label)
        cmap          : colormap name
        tooltip_fields: list of column names to show on hover

    Returns:
        folium.Map
    """
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    # Build column list: base cols + any extra tooltip fields that exist in gdf
    base_cols = ["ubigeo", "distrito", "departamen", column, "geometry"]
    extra_cols = [f for f in (tooltip_fields or []) if f in gdf.columns and f not in base_cols]
    gdf_json = gdf[base_cols + extra_cols].copy()
    gdf_json = gdf_json.dropna(subset=[column])

    folium.Choropleth(
        geo_data=gdf_json.__geo_interface__,
        data=gdf_json,
        columns=["ubigeo", column],
        key_on="feature.properties.ubigeo",
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        name=title,
    ).add_to(m)

    if tooltip_fields is None:
        tooltip_fields = ["distrito", "departamen", column]

    tooltip_fields = [f for f in tooltip_fields if f in gdf_json.columns]
    folium.GeoJson(
        gdf_json.__geo_interface__,
        style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=GeoJsonTooltip(fields=tooltip_fields, localize=True),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def folium_classification_map(gdf):
    """
    Build a Folium map with 3-tier classification coloring.

    Returns folium.Map
    """
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    gdf_copy = gdf[["ubigeo", "distrito", "departamen", "class_baseline",
                     "score_baseline", "n_ipress", "mean_dist_km",
                     "geometry"]].copy()

    def style_fn(feature):
        cls = feature["properties"].get("class_baseline", "Average")
        return {
            "fillColor":   CLASS_COLORS.get(cls, "#fee090"),
            "color":       "grey",
            "weight":      0.3,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        gdf_copy.__geo_interface__,
        style_function=style_fn,
        tooltip=GeoJsonTooltip(
            fields=["distrito", "departamen", "class_baseline",
                    "score_baseline", "n_ipress", "mean_dist_km"],
            aliases=["District", "Department", "Classification",
                     "Score", "# IPRESS", "Mean dist (km)"],
            localize=True,
        ),
        name="Classification",
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px; border-radius:5px;
                border:1px solid grey; font-size:13px;">
      <b>Classification</b><br>
      <span style="background:#d73027;padding:2px 8px;">&nbsp;</span> Underserved<br>
      <span style="background:#fee090;padding:2px 8px;">&nbsp;</span> Average<br>
      <span style="background:#1a9850;padding:2px 8px;">&nbsp;</span> Well-served
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


def folium_ipress_points(gdf, ipress_df):
    """
    Build a Folium map with IPRESS facility points overlaid on district boundaries.

    Parameters:
        gdf      : district GeoDataFrame
        ipress_df: IPRESS spatial DataFrame with lon/lat columns

    Returns folium.Map
    """
    center = [-9.19, -75.01]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    # District boundaries (light)
    folium.GeoJson(
        gdf[["ubigeo", "distrito", "geometry"]].__geo_interface__,
        style_function=lambda f: {
            "fillColor": "none", "color": "#aaaaaa", "weight": 0.4
        },
    ).add_to(m)

    # IPRESS points (clustered for performance)
    from folium.plugins import MarkerCluster
    cluster = MarkerCluster(name="IPRESS Facilities").add_to(m)

    sample = ipress_df.dropna(subset=["lon", "lat"]).head(5000)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#2171b5",
            fill=True,
            fill_opacity=0.7,
            popup="{}<br>Cat: {}<br>Sector: {}".format(
                row.get("nombre_del_establecimiento", ""),
                row.get("categoria", ""),
                row.get("institucion", ""),
            ),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


def map_to_html(folium_map):
    """Return Folium map as HTML string for Streamlit rendering."""
    return folium_map._repr_html_()


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_number(n):
    """Format integer with thousands separator."""
    return "{:,}".format(int(n))


def fmt_pct(v):
    """Format float as percentage string."""
    return "{:.1f}%".format(v)


def fmt_km(v):
    """Format float as km string."""
    return "{:.2f} km".format(v)
