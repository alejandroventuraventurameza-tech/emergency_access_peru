# -*- coding: utf-8 -*-
"""
data_loader.py - Emergency Healthcare Access Inequality in Peru

Responsible for loading all raw datasets into clean Python objects
ready for preprocessing. No transformation logic lives here.
"""

import os
import geopandas as gpd
import pandas as pd

# Paths
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

PATHS = {
    "ccpp":            os.path.join(RAW_DIR, "CCPP_0", "CCPP_IGN100K.shp"),
    "distritos":       os.path.join(RAW_DIR, "DISTRITOS.shp"),
    "ipress":          os.path.join(RAW_DIR, "IPRESS.csv"),
    "emergencia_2024": os.path.join(RAW_DIR, "ConsultaC1_2024_v22.csv"),
    "emergencia_2025": os.path.join(RAW_DIR, "ConsultaC1_2025_v20.csv"),
}


def load_ccpp():
    """
    Load the Centros Poblados shapefile (IGN 1:100K).

    Returns GeoDataFrame with 136K+ populated centers.
    CRS: EPSG:4326 (WGS84).

    Key columns:
        NOM_POBLAD : place name
        DIST       : district name
        PROV       : province name
        DEP        : department name
        COD_INT    : internal code
        CATEGORIA  : settlement category
        X, Y       : longitude / latitude
        geometry   : Point geometry
    """
    gdf = gpd.read_file(PATHS["ccpp"])
    return gdf


def load_distritos():
    """
    Load the district boundary shapefile.

    Returns GeoDataFrame with 1,873 Peruvian districts.
    CRS: EPSG:4326 (WGS84).

    Key columns:
        IDDPTO     : department code (2-digit)
        DEPARTAMEN : department name
        IDPROV     : province code (4-digit)
        PROVINCIA  : province name
        IDDIST     : district code (6-digit, equivalent to UBIGEO)
        DISTRITO   : district name
        CODCCPP    : district capital code
        AREA       : area in native units
        geometry   : Polygon geometry
    """
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(PATHS["distritos"])
    return gdf


def load_ipress():
    """
    Load the MINSA IPRESS health facilities registry.

    Returns DataFrame with all registered health facilities.
    Encoding: latin-1.

    Key columns:
        Institucion               : sector (MINSA, ESSALUD, PRIVADO, etc.)
        Codigo Unico              : facility unique ID (matches CO_IPRESS)
        Nombre del establecimiento: facility name
        UBIGEO                    : district UBIGEO code (6-digit string)
        Departamento/Provincia/Distrito : location labels
        Categoria                 : MINSA level (I-1 to III-2)
        Estado                    : ACTIVADO / DESACTIVADO
        NORTE / ESTE              : coordinates (latitude/longitude)
        CAMAS                     : number of beds
    """
    df = pd.read_csv(
        PATHS["ipress"],
        encoding="latin-1",
        dtype={"UBIGEO": str},
        low_memory=False,
    )
    return df


def load_emergencia(year=2024):
    """
    Load the emergency care production dataset.
    (Produccion Asistencial en Emergencia por IPRESS)

    Parameters:
        year : int - 2024 or 2025

    Returns DataFrame with monthly emergency attention records by IPRESS.
    Separator: semicolon. Encoding: latin-1.

    Key columns:
        ANHO                 : year
        MES                  : month (01-12)
        UBIGEO               : district UBIGEO (6-digit string)
        DEPARTAMENTO/PROVINCIA/DISTRITO : location labels
        SECTOR               : sector
        CATEGORIA            : IPRESS level (I-1 to III-2)
        CO_IPRESS            : facility code
        RAZON_SOC            : facility name
        NRO_TOTAL_ATENCIONES : total emergency attendances
        NRO_TOTAL_ATENDIDOS  : total emergency patients seen
    """
    key = "emergencia_{}".format(year)
    if key not in PATHS:
        raise ValueError("Year {} not available. Choose 2024 or 2025.".format(year))

    df = pd.read_csv(
        PATHS[key],
        sep=";",
        encoding="latin-1",
        dtype={
            "UBIGEO": str,
            "CO_IPRESS": str,
            "NRO_TOTAL_ATENCIONES": str,
            "NRO_TOTAL_ATENDIDOS": str,
        },
        low_memory=False,
    )
    return df


def load_all():
    """
    Convenience function: load all datasets at once.

    Returns dict with keys:
        'ccpp', 'distritos', 'ipress', 'emergencia_2024', 'emergencia_2025'
    """
    return {
        "ccpp":            load_ccpp(),
        "distritos":       load_distritos(),
        "ipress":          load_ipress(),
        "emergencia_2024": load_emergencia(2024),
        "emergencia_2025": load_emergencia(2025),
    }


if __name__ == "__main__":
    datasets = load_all()
    for name, obj in datasets.items():
        print("{:25s} -> shape: {}  | type: {}".format(
            name, obj.shape, type(obj).__name__))
