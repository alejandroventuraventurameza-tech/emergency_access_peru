# -*- coding: utf-8 -*-
"""
cleaning.py - Emergency Healthcare Access Inequality in Peru

Cleans and preprocesses each raw dataset. Outputs are saved to
data/processed/. No spatial joins or metric construction here.

Key decisions documented per dataset.
"""

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_col(name):
    """Lowercase, strip accents, replace spaces/dots with underscore."""
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ü': 'u', 'ñ': 'n', 'Ñ': 'N',
    }
    for src, dst in replacements.items():
        name = name.replace(src, dst)
    name = re.sub(r'[\s\.\-\/]+', '_', name)
    name = re.sub(r'[^\w]', '', name)
    return name.lower().strip('_')


def normalize_columns(df):
    """Return df with normalized column names."""
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    return df


def pad_ubigeo(series, width=6):
    """Zero-pad UBIGEO codes to 6 digits."""
    return series.astype(str).str.zfill(width)


# ── IPRESS ────────────────────────────────────────────────────────────────────

def clean_ipress(df):
    """
    Clean the MINSA IPRESS health facilities dataset.

    Decisions:
    - Standardize column names (remove accents, lowercase).
    - Rename NORTE -> lon, ESTE -> lat. In the raw file the columns are
      labeled NORTE/ESTE but the values confirm NORTE holds longitude
      (~-68 to -81) and ESTE holds latitude (~-18 to 0). We rename them
      to avoid confusion throughout the pipeline.
    - Keep only ACTIVADO facilities (all 20,819 rows are ACTIVADO,
      so no rows are dropped by this filter but the logic is explicit).
    - Drop rows without valid coordinates (12,863 rows have null coords).
      These facilities cannot be used in spatial analysis. They are
      retained in a separate subset for attribute-only analysis.
    - Drop rows where coordinates are exactly 0 (3 rows — invalid).
    - Zero-pad UBIGEO to 6 digits.
    - Save two outputs:
        ipress_clean.csv     -> all ACTIVADO facilities (with & without coords)
        ipress_spatial.csv   -> subset with valid coordinates only
    """
    df = normalize_columns(df)

    # Rename coordinate columns (values confirm the swap)
    df = df.rename(columns={"norte": "lon", "este": "lat"})

    # Keep only active facilities
    df = df[df["estado"] == "ACTIVADO"].copy()

    # Zero-pad UBIGEO
    df["ubigeo"] = pad_ubigeo(df["ubigeo"])

    # Flag valid coordinates
    df["has_coords"] = (
        df["lon"].notna() & df["lat"].notna() &
        (df["lon"] != 0) & (df["lat"] != 0)
    )

    # Spatial subset
    ipress_spatial = df[df["has_coords"]].copy()

    print("[IPRESS] Total ACTIVADO: {:,}".format(len(df)))
    print("[IPRESS] With valid coords: {:,}  |  Without: {:,}".format(
        len(ipress_spatial), len(df) - len(ipress_spatial)))

    # Save
    df.to_csv(os.path.join(PROCESSED_DIR, "ipress_clean.csv"), index=False)
    ipress_spatial.to_csv(os.path.join(PROCESSED_DIR, "ipress_spatial.csv"), index=False)

    return df, ipress_spatial


# ── EMERGENCIA ────────────────────────────────────────────────────────────────

def clean_emergencia(df, year):
    """
    Clean one year of emergency care production data.

    Decisions:
    - NRO_TOTAL_ATENCIONES and NRO_TOTAL_ATENDIDOS contain 'NE_XXXX'
      placeholder codes. These represent suppressed values (privacy
      protection for small counts). We replace them with NaN and convert
      to numeric. Rows with NaN are excluded from district aggregations.
    - UBIGEO zero-padded to 6 digits.
    - Aggregate to district level: sum atenciones and atendidos,
      count distinct IPRESS per district.
    - Save:
        emergencia_{year}_clean.csv   -> row-level cleaned data
        emergencia_{year}_district.csv -> district-level aggregation
    """
    df = df.copy()

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]

    # Zero-pad UBIGEO
    df["ubigeo"] = pad_ubigeo(df["ubigeo"])

    # Replace NE_ placeholder values with NaN
    for col in ["nro_total_atenciones", "nro_total_atendidos"]:
        ne_mask = df[col].astype(str).str.startswith("NE_", na=False)
        df.loc[ne_mask, col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ne_count = df["nro_total_atenciones"].isna().sum()
    print("[EMERGENCIA {}] NE_ rows set to NaN: {:,} / {:,}".format(
        year, ne_count, len(df)))

    # District-level aggregation
    district = (
        df.dropna(subset=["nro_total_atenciones"])
        .groupby("ubigeo", as_index=False)
        .agg(
            departamento=("departamento", "first"),
            provincia=("provincia", "first"),
            distrito=("distrito", "first"),
            n_ipress=("co_ipress", "nunique"),
            total_atenciones=("nro_total_atenciones", "sum"),
            total_atendidos=("nro_total_atendidos", "sum"),
        )
    )

    print("[EMERGENCIA {}] Districts with data: {:,}".format(year, len(district)))

    # Save
    df.to_csv(os.path.join(PROCESSED_DIR, "emergencia_{}_clean.csv".format(year)),
              index=False)
    district.to_csv(os.path.join(PROCESSED_DIR, "emergencia_{}_district.csv".format(year)),
                    index=False)

    return df, district


# ── CCPP ─────────────────────────────────────────────────────────────────────

def clean_ccpp(gdf):
    """
    Clean the Centros Poblados GeoDataFrame.

    Decisions:
    - Normalize column names.
    - Drop rows with null geometry (none found, but guard is explicit).
    - Keep only relevant columns to reduce memory footprint.
    - CCPP does not carry a UBIGEO code directly; district assignment
      will be done via spatial join in geospatial.py.
    - Save: ccpp_clean.gpkg (GeoPackage preserves geometry + CRS).
    """
    gdf = gdf.copy()
    gdf = normalize_columns(gdf)

    # Rename columns with unicode artifacts
    rename_map = {}
    for c in gdf.columns:
        if 'c_d_int' in c or 'cod_int' in c or 'c_d' in c:
            rename_map[c] = 'cod_int'
        if 'c_digo' in c or 'codigo' in c and c != 'cod_int':
            rename_map[c] = 'codigo'
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    # Drop null geometry
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    print("[CCPP] Dropped null/empty geometry: {:,}".format(before - len(gdf)))
    print("[CCPP] Remaining: {:,}  |  CRS: {}".format(len(gdf), gdf.crs))

    # Keep relevant columns
    keep = [c for c in ['nom_poblad', 'cat_poblad', 'categoria', 'dist',
                         'prov', 'dep', 'x', 'y', 'geometry'] if c in gdf.columns]
    gdf = gdf[keep]

    # Save
    out = os.path.join(PROCESSED_DIR, "ccpp_clean.parquet")
    gdf.to_parquet(out)
    print("[CCPP] Saved to ccpp_clean.parquet")

    return gdf


# ── DISTRITOS ─────────────────────────────────────────────────────────────────

def clean_distritos(gdf):
    """
    Clean the district boundaries GeoDataFrame.

    Decisions:
    - Normalize column names.
    - Rename IDDIST -> ubigeo for consistency across all datasets.
    - CAPITAL has 1 null row — not critical, kept as-is.
    - Drop rows with null or empty geometry (none found).
    - CRS is EPSG:4326 — confirmed correct for Peru national coverage.
    - Save: distritos_clean.gpkg
    """
    gdf = gdf.copy()
    gdf = normalize_columns(gdf)

    # Rename for pipeline consistency
    gdf = gdf.rename(columns={"iddist": "ubigeo"})

    # Zero-pad UBIGEO
    gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])

    # Drop null/empty geometry
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    print("[DISTRITOS] Dropped null/empty geometry: {:,}".format(before - len(gdf)))
    print("[DISTRITOS] Remaining: {:,}  |  CRS: {}".format(len(gdf), gdf.crs))

    # Save
    out = os.path.join(PROCESSED_DIR, "distritos_clean.parquet")
    gdf.to_parquet(out)
    print("[DISTRITOS] Saved to distritos_clean.parquet")

    return gdf


# ── Run all ───────────────────────────────────────────────────────────────────

def clean_all(datasets):
    """
    Run all cleaning functions.

    Parameters:
        datasets : dict returned by data_loader.load_all()

    Returns:
        dict with cleaned versions of all datasets
    """
    print("\n--- Cleaning IPRESS ---")
    ipress_clean, ipress_spatial = clean_ipress(datasets["ipress"])

    print("\n--- Cleaning EMERGENCIA 2024 ---")
    em2024_clean, em2024_district = clean_emergencia(datasets["emergencia_2024"], 2024)

    print("\n--- Cleaning EMERGENCIA 2025 ---")
    em2025_clean, em2025_district = clean_emergencia(datasets["emergencia_2025"], 2025)

    print("\n--- Cleaning CCPP ---")
    ccpp_clean = clean_ccpp(datasets["ccpp"])

    print("\n--- Cleaning DISTRITOS ---")
    distritos_clean = clean_distritos(datasets["distritos"])

    print("\n[OK] All cleaned datasets saved to data/processed/")

    return {
        "ipress":          ipress_clean,
        "ipress_spatial":  ipress_spatial,
        "em2024":          em2024_clean,
        "em2024_district": em2024_district,
        "em2025":          em2025_clean,
        "em2025_district": em2025_district,
        "ccpp":            ccpp_clean,
        "distritos":       distritos_clean,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all
    datasets = load_all()
    clean_all(datasets)
