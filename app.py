"""
app.py — Streamlit application
Emergency Healthcare Access Inequality in Peru
"""
import streamlit as st

st.set_page_config(page_title="Emergency Healthcare Access — Peru", layout="wide")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ GeoSpatial Results",
    "🔍 Interactive Exploration",
])

with tab1:
    st.header("Data & Methodology")
    st.write("Problem statement, data sources, cleaning summary, and methodological decisions.")

with tab2:
    st.header("Static Analysis")
    st.write("Selected charts with interpretations.")

with tab3:
    st.header("GeoSpatial Results")
    st.write("Static maps, district-level comparisons, and supporting tables.")

with tab4:
    st.header("Interactive Exploration")
    st.write("Folium maps, district comparison views, baseline vs alternative.")
