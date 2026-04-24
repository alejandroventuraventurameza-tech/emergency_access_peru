# -*- coding: utf-8 -*-
"""
geospatial.py - Emergency Healthcare Access Inequality in Peru

Builds all spatial relationships between datasets:
  1. Creates a GeoDataFrame from IPRESS facilities (point layer).
  2. Assigns IPRESS facilities to districts via spatial join.
  3. Assigns CCPP populated centers to districts via spatial join.
  4. Computes nearest-IPRESS distance for each populated center.
  5. Produces a district-level spatial summary table.

CRS strategy:
  - All input data arrives in EPSG:4326 (WGS84 geographic).
  - Distance calculations are performed in EPSG:32718
    (WGS 84 / UTM zone 18S), which covers Peru's main populated
    zones and minimizes distance distortion for national analysis.
    This is a deliberate methodological choice: UTM 18S is the
    official projection used by IGN Peru for 1:100K cartography.
  - All outputs are reprojected back to EPSG:4326 for compatibility
    with Folium and downstream visualization.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Projected CRS for distance calculations (meters)
CRS_PROJECTED = "EPSG:32718"
CRS_GEO = "EPSG:4326"


# ── 1. Build IPRESS GeoDataFrame ──────────────────────────────────────────────

def make_ipress_gdf(ipress_spatial_df):
    """
    Convert IPRESS spatial CSV (with lon/lat) to a GeoDataFrame.

    Parameters:
        ipress_spatial_df : pd.DataFrame from ipress_spatial.csv

    Returns:
        GeoDataFrame with Point geometry, CRS EPSG:4326.
    """
    gdf = gpd.GeoDataFrame(
        ipress_spatial_df,
        geometry=gpd.points_from_xy(
            ipress_spatial_df["lon"],
            ipress_spatial_df["lat"],
        ),
        crs=CRS_GEO,
    )
    return gdf


# ── 2. Assign IPRESS to districts ─────────────────────────────────────────────

def assign_ipress_to_districts(ipress_gdf, distritos_gdf):
    """
    Spatial join: assign each IPRESS facility to the district polygon
    it falls within.

    Decision: we use the UBIGEO already present in the IPRESS registry
    as the primary district key, but verify it against the spatial join.
    Facilities that fall outside all district polygons (e.g., border
    artifacts) retain their registry UBIGEO.

    Parameters:
        ipress_gdf    : GeoDataFrame (IPRESS points, EPSG:4326)
        distritos_gdf : GeoDataFrame (district polygons, EPSG:4326)

    Returns:
        GeoDataFrame with 'ubigeo_spatial' column added.
    """
    # Keep only needed district columns for join
    dist_slim = distritos_gdf[["ubigeo", "distrito", "geometry"]].rename(
        columns={"ubigeo": "ubigeo_spatial", "distrito": "distrito_spatial"}
    )

    joined = gpd.sjoin(
        ipress_gdf,
        dist_slim,
        how="left",
        predicate="within",
    )
    # Drop the index_right artifact from sjoin
    joined = joined.drop(columns=["index_right"], errors="ignore")

    n_matched = joined["ubigeo_spatial"].notna().sum()
    print("[IPRESS->DIST] Spatially matched: {:,} / {:,}".format(
        n_matched, len(joined)))

    return joined


# ── 3. Assign CCPP to districts ───────────────────────────────────────────────

def assign_ccpp_to_districts(ccpp_gdf, distritos_gdf):
    """
    Spatial join: assign each populated center to its district polygon.

    CCPP does not carry a UBIGEO code, only text names (DIST, PROV, DEP).
    The spatial join is the authoritative assignment method.

    Parameters:
        ccpp_gdf      : GeoDataFrame (CCPP points, EPSG:4326)
        distritos_gdf : GeoDataFrame (district polygons, EPSG:4326)

    Returns:
        GeoDataFrame with 'ubigeo', 'distrito', 'departamen', 'provincia'
        columns added from the district layer.
    """
    dist_slim = distritos_gdf[[
        "ubigeo", "distrito", "departamen", "provincia", "geometry"
    ]].rename(columns={"departamen": "dep_dist", "provincia": "prov_dist"})

    joined = gpd.sjoin(
        ccpp_gdf,
        dist_slim,
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")

    n_matched = joined["ubigeo"].notna().sum()
    n_unmatched = joined["ubigeo"].isna().sum()
    print("[CCPP->DIST] Matched: {:,}  |  Unmatched (border): {:,}".format(
        n_matched, n_unmatched))

    return joined


# ── 4. Nearest IPRESS distance for each CCPP ─────────────────────────────────

def compute_nearest_ipress_distance(ccpp_gdf, ipress_gdf):
    """
    For each populated center (CCPP), find the nearest IPRESS facility
    and compute the straight-line distance in kilometers.

    Method: scipy cKDTree on projected coordinates (EPSG:32718, meters).
    Straight-line distance is used as a proxy for access — a standard
    approach in health geography when road network data is unavailable.
    This is explicitly noted as a limitation in the methodology.

    Parameters:
        ccpp_gdf   : GeoDataFrame (CCPP points, any CRS)
        ipress_gdf : GeoDataFrame (IPRESS points, any CRS)

    Returns:
        ccpp_gdf with two new columns:
            nearest_ipress_dist_km  : distance to nearest facility (km)
            nearest_ipress_code     : Codigo Unico of nearest facility
    """
    # Project both layers to UTM 18S
    ccpp_proj  = ccpp_gdf.to_crs(CRS_PROJECTED)
    ipress_proj = ipress_gdf.to_crs(CRS_PROJECTED)

    ccpp_coords  = np.array(list(zip(
        ccpp_proj.geometry.x, ccpp_proj.geometry.y)))
    ipress_coords = np.array(list(zip(
        ipress_proj.geometry.x, ipress_proj.geometry.y)))

    tree = cKDTree(ipress_coords)
    dist_m, idx = tree.query(ccpp_coords, k=1)

    result = ccpp_gdf.copy()
    result["nearest_ipress_dist_km"] = dist_m / 1000.0
    result["nearest_ipress_code"] = ipress_proj.iloc[idx]["codigo_unico"].values

    print("[CCPP->IPRESS] Mean nearest distance: {:.2f} km".format(
        result["nearest_ipress_dist_km"].mean()))
    print("[CCPP->IPRESS] Median nearest distance: {:.2f} km".format(
        result["nearest_ipress_dist_km"].median()))
    print("[CCPP->IPRESS] CCPP within 5 km of IPRESS: {:.1f}%".format(
        (result["nearest_ipress_dist_km"] <= 5).mean() * 100))

    return result


# ── 5. District-level spatial summary ────────────────────────────────────────

def build_district_spatial_summary(ccpp_with_dist, ipress_with_dist, distritos_gdf):
    """
    Aggregate spatial relationships to the district level.

    Produces one row per district with:
        n_ipress            : number of IPRESS facilities in district
        n_ipress_categories : number of distinct IPRESS categories
        n_ccpp              : number of populated centers in district
        mean_dist_km        : mean distance CCPP->nearest IPRESS (km)
        median_dist_km      : median distance CCPP->nearest IPRESS (km)
        pct_ccpp_within5km  : % of CCPP within 5 km of any IPRESS
        pct_ccpp_within10km : % of CCPP within 10 km of any IPRESS

    Parameters:
        ccpp_with_dist   : output of compute_nearest_ipress_distance()
        ipress_with_dist : output of assign_ipress_to_districts()
        distritos_gdf    : clean district GeoDataFrame

    Returns:
        GeoDataFrame: district polygons merged with spatial summary.
    """
    # IPRESS aggregation per district
    ipress_agg = (
        ipress_with_dist
        .groupby("ubigeo", as_index=False)
        .agg(
            n_ipress=("codigo_unico", "count"),
            n_ipress_cat=("categoria", "nunique"),
        )
    )

    # CCPP aggregation per district
    ccpp_agg = (
        ccpp_with_dist
        .groupby("ubigeo", as_index=False)
        .agg(
            n_ccpp=("nom_poblad", "count"),
            mean_dist_km=("nearest_ipress_dist_km", "mean"),
            median_dist_km=("nearest_ipress_dist_km", "median"),
            pct_ccpp_within5km=(
                "nearest_ipress_dist_km",
                lambda x: (x <= 5).mean() * 100
            ),
            pct_ccpp_within10km=(
                "nearest_ipress_dist_km",
                lambda x: (x <= 10).mean() * 100
            ),
        )
    )

    # Merge with district polygons
    summary = distritos_gdf.merge(ipress_agg, on="ubigeo", how="left")
    summary = summary.merge(ccpp_agg, on="ubigeo", how="left")

    # Fill districts with no facilities or CCPP with 0
    for col in ["n_ipress", "n_ipress_cat", "n_ccpp"]:
        summary[col] = summary[col].fillna(0).astype(int)

    n_no_ipress = (summary["n_ipress"] == 0).sum()
    n_no_ccpp = (summary["n_ccpp"] == 0).sum()
    print("[DIST SUMMARY] Districts with no IPRESS: {:,}".format(n_no_ipress))
    print("[DIST SUMMARY] Districts with no CCPP: {:,}".format(n_no_ccpp))
    print("[DIST SUMMARY] Total districts: {:,}".format(len(summary)))

    return summary


# ── Run all ───────────────────────────────────────────────────────────────────

def build_geospatial_pipeline(cleaned):
    """
    Run the full geospatial pipeline.

    Parameters:
        cleaned : dict returned by cleaning.clean_all()

    Returns:
        dict with:
            'ipress_gdf'       : IPRESS as GeoDataFrame
            'ipress_district'  : IPRESS with district assignment
            'ccpp_district'    : CCPP with district assignment
            'ccpp_distances'   : CCPP with nearest IPRESS distance
            'district_spatial' : district-level spatial summary GeoDataFrame
    """
    print("\n--- Building IPRESS GeoDataFrame ---")
    ipress_spatial = pd.read_csv(
        os.path.join(PROCESSED_DIR, "ipress_spatial.csv"),
        dtype={"ubigeo": str},
    )
    ipress_gdf = make_ipress_gdf(ipress_spatial)

    distritos = cleaned["distritos"]
    ccpp = cleaned["ccpp"]

    print("\n--- Assigning IPRESS to districts ---")
    ipress_district = assign_ipress_to_districts(ipress_gdf, distritos)

    print("\n--- Assigning CCPP to districts ---")
    ccpp_district = assign_ccpp_to_districts(ccpp, distritos)

    print("\n--- Computing CCPP -> nearest IPRESS distances ---")
    ccpp_distances = compute_nearest_ipress_distance(ccpp_district, ipress_gdf)

    print("\n--- Building district spatial summary ---")
    district_spatial = build_district_spatial_summary(
        ccpp_distances, ipress_district, distritos
    )

    # Save district spatial summary
    out = os.path.join(PROCESSED_DIR, "district_spatial.parquet")
    district_spatial.to_parquet(out)
    print("[OK] Saved district_spatial.parquet")

    return {
        "ipress_gdf":       ipress_gdf,
        "ipress_district":  ipress_district,
        "ccpp_district":    ccpp_district,
        "ccpp_distances":   ccpp_distances,
        "district_spatial": district_spatial,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cleaning import clean_all
    from data_loader import load_all

    datasets = load_all()
    cleaned = clean_all(datasets)
    results = build_geospatial_pipeline(cleaned)

    ds = results["district_spatial"]
    print("\nDistrict spatial summary sample:")
    print(ds[["ubigeo", "distrito", "n_ipress", "n_ccpp",
               "mean_dist_km", "pct_ccpp_within5km"]].head(10).to_string(index=False))
