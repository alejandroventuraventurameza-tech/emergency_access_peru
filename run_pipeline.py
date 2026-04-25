# -*- coding: utf-8 -*-
"""
run_pipeline.py - End-to-end pipeline runner.
Executes all processing steps in order and reports status.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def step(name, fn):
    print("\n" + "="*60)
    print("STEP: {}".format(name))
    print("="*60)
    t0 = time.time()
    result = fn()
    print("[OK] {} completed in {:.1f}s".format(name, time.time() - t0))
    return result

# Step 1: Load raw data
from data_loader import load_all
datasets = step("Load raw datasets", load_all)

# Step 2: Clean
from cleaning import clean_all
cleaned = step("Clean all datasets", lambda: clean_all(datasets))

# Step 3: Geospatial pipeline
from geospatial import build_geospatial_pipeline
geo = step("Geospatial pipeline", lambda: build_geospatial_pipeline(cleaned))

# Step 4: Metrics
from metrics import build_metrics
metrics = step("Build metrics", build_metrics)

# Step 5: Visualizations
from visualization import generate_all
step("Generate visualizations", generate_all)

# Summary
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
df = metrics["district_scores"]
print("Districts analyzed:    {:,}".format(len(df)))
print("Underserved:           {:,}".format((df['class_baseline']=='Underserved').sum()))
print("Well-served:           {:,}".format((df['class_baseline']=='Well-served').sum()))
print("Spearman rho (specs):  {:.4f}".format(metrics["spearman_rho"]))
print("Figures generated:     {:,}".format(
    len([f for f in os.listdir("output/figures") if f.endswith(".png")])))
print("Processed files:       {:,}".format(
    len(os.listdir("data/processed"))))
