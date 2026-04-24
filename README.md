# Emergency Healthcare Access Inequality in Peru

**Course:** Python Programming / Data Science  
**Assignment:** HW_02_202601  
**Deadline:** Friday, April 24 — 11:59 PM

---

## What does the project do?

This project builds a complete geospatial analytics pipeline in Python to study **emergency healthcare access inequality across districts in Peru**. It combines four public datasets into a district-level analytical framework that ranks and classifies all 1,873 Peruvian districts by their access to emergency health services.

---

## Main Analytical Goal

Answer the following question with quantitative evidence:

> **Which districts in Peru appear relatively better or worse served in emergency healthcare access, and what evidence supports that conclusion?**

The analysis is deliberately methodological — it is not simply a map of hospitals. It designs, justifies, and stress-tests a district-level access index built from facility availability, emergency care activity, and the spatial relationship between populated centers and health facilities.

---

## Datasets Used

| Dataset | Source | Records | File |
|---|---|---|---|
| Centros Poblados (CCPP) | IGN 1:100K | 136,587 | `data/raw/CCPP_0/CCPP_IGN100K.shp` |
| District Boundaries | IGN / MINSA | 1,873 | `data/raw/DISTRITOS.shp` |
| IPRESS Health Facilities | MINSA | 20,819 | `data/raw/IPRESS.csv` |
| Emergency Production 2024 | MINSA | 250,000 | `data/raw/ConsultaC1_2024_v22.csv` |
| Emergency Production 2025 | MINSA | 342,753 | `data/raw/ConsultaC1_2025_v20.csv` |

---

## Data Cleaning Decisions

**IPRESS:**
- Column names standardized (accents removed, lowercase, underscores).
- Coordinate columns were **swapped** in the raw file: `NORTE` held longitude values (~-68 to -81) and `ESTE` held latitude (~-18 to 0). Renamed to `lon` / `lat`.
- 12,866 facilities lacked coordinates — excluded from spatial analysis, retained for attribute analysis.

**Emergency data (2024 / 2025):**
- `NE_XXXX` codes represent suppressed counts (privacy protection for small facilities). Replaced with `NaN` and excluded from aggregations. 2024: 36,377 rows; 2025: 44,889 rows.
- Aggregated to district level: sum of atenciones, atendidos; count of distinct IPRESS.

**CCPP:**
- No null geometries found. District assignment done via **spatial join** (not name matching) — authoritative and independent of spelling inconsistencies.
- 221 centers (< 0.2%) fell on district borders; left unmatched.

**DISTRITOS:**
- `IDDIST` renamed to `ubigeo` for pipeline consistency. Zero-padded to 6 digits throughout.

---

## District-Level Metrics

### Baseline Specification (equal weights, 2024 data, 5 km threshold)

All components min-max normalized to [0, 1]:

| Component | Formula |
|---|---|
| Facility availability | `n_ipress / n_ccpp` per district |
| Emergency activity | `total_atenciones_2024 / n_ipress` |
| Spatial access | `% of CCPP within 5 km of any IPRESS` |

**Score** = (facility + activity + access) / 3

### Alternative Specification (weighted, 2025 data, 10 km threshold)

| Component | Formula | Weight |
|---|---|---|
| Facility density | `n_ipress / area_km2` | 25% |
| Patient throughput | `total_atendidos_2025 / n_ccpp` | 25% |
| Broad spatial access | `% of CCPP within 10 km` | 50% |

**Rationale for alternative weights:** Spatial coverage is a more fundamental equity criterion — ensuring *any* facility is reachable matters more than facility density per se. The 10 km threshold is more appropriate for highland and jungle districts where 5 km coverage is structurally unattainable.

### CRS Strategy

- All inputs in **EPSG:4326** (WGS84 geographic).
- Distance calculations in **EPSG:32718** (WGS84 / UTM zone 18S) — IGN Peru standard for 1:100K cartography, minimizes distance distortion for national analysis.
- All outputs returned in EPSG:4326 for Folium compatibility.

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `geopandas`, `folium`, `matplotlib`, `seaborn`, `streamlit`, `pandas`, `numpy`, `shapely`, `pyproj`, `scipy`, `pyarrow`

---

## How to Run the Pipeline

Run each module in order, or run them individually:

```bash
# Step 1: Load and inspect raw data
python src/data_loader.py

# Step 2: Clean and save to data/processed/
python src/cleaning.py

# Step 3: Spatial joins and distance calculations
python src/geospatial.py

# Step 4: Build district-level metrics (both specifications)
python src/metrics.py

# Step 5: Generate all static charts and maps
python src/visualization.py
```

---

## How to Run the Streamlit App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with 4 tabs:
1. **Data & Methodology** — problem statement, data sources, cleaning decisions, limitations
2. **Static Analysis** — 9 charts answering all 4 analytical questions
3. **GeoSpatial Results** — 4 static maps + filterable district table
4. **Interactive Exploration** — 4 Folium maps + specification comparison

---

## Main Findings

- **562 districts (30%) classified as Underserved** — concentrated in the highlands (Puno, Apurímac, Cusco) and Amazon jungle.
- **563 districts (30%) classified as Well-served** — concentrated in Lima metropolitan area, major coastal cities (Arequipa, Trujillo, Chiclayo), and provincial capitals.
- **Best-served districts (baseline):** Yanahuara (Arequipa), Lima, Villa El Salvador, Arequipa, Rímac.
- **Most underserved districts:** Rural districts in Puno and Arequipa with 0 IPRESS facilities and mean distances exceeding 30–70 km.
- **Spatial access nationally:** Median distance from populated centers to nearest IPRESS = 3.5 km; 64.5% of CCPP within 5 km.
- **Specification sensitivity (Q4):** Spearman rho = 0.888 between both specs. 471 of 1,873 districts change tier — mostly peri-urban and mid-sized highland districts sensitive to the 5 km vs 10 km threshold choice.

---

## Main Limitations

1. **Straight-line distance** used as access proxy (road network data unavailable at national scale). Underestimates true travel time, especially in Andes terrain.
2. **12,866 IPRESS without coordinates** excluded from spatial analysis — true facility counts per district are higher than reported.
3. **Emergency data covers only reporting IPRESS** — silent facilities may be inactive or simply non-reporting.
4. **Population proxy:** CCPP count used as settlement proxy (true population data not integrated). Large cities may be under-penalized relative to actual population burden.
5. **10 UBIGEO codes** in emergency data do not match the district shapefile — likely new districts created after shapefile publication.
6. **UTM 18S projection** optimized for central Peru — introduces some distortion for far-western (Tumbes) and far-eastern (Madre de Dios) districts.

---

## Repository Structure

```
emergency_access_peru/
│
├── app.py                    # Streamlit app (4 tabs)
├── run_pipeline.py           # End-to-end pipeline runner
├── README.md
├── requirements.txt
│
├── src/
│   ├── data_loader.py        # Raw data loading functions
│   ├── cleaning.py           # Cleaning and preprocessing
│   ├── geospatial.py         # Spatial joins, distance, GeoDataFrames
│   ├── metrics.py            # District-level scores (baseline + alternative)
│   ├── visualization.py      # Static charts and maps
│   └── utils.py              # Shared helpers and Folium builders
│
├── data/
│   ├── raw/                  # Original downloaded files
│   └── processed/            # Cleaned and processed outputs (parquet / csv)
│
├── output/
│   ├── figures/              # 9 static charts + 4 maps (PNG)
│   └── tables/               # district_scores.csv, spec_comparison.csv
│
└── video/
    └── link.txt              # Link to explanatory video
```
