#!/usr/bin/env python

"""
SCRIPT 2: CHART GENERATOR (Using chargers_locations.geojson)

This script collects data from chargers_locations.geojson, runs simple
simulations (A, B, C), and generates all charts.

It REQUIRES the files 'chargers_locations.geojson' and
'nsga2_all_k_results.csv' to be present.
"""

import os
import json
import math
import random
import glob
import sys
from pathlib import Path
from datetime import datetime, timezone # Import timezone for comparison

# Core data handling and analysis
import pandas as pd
import numpy as np
import geopandas as gpd
import requests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# --- 0. Global Setup ---
OUTPUT_DIR = Path("./simulation_outputs_final_geojson")
NSGA2_FILE_PATH = "nsga2_all_k_results_GENERATED.csv" # Updated to match generator output
CHARGERS_GEOJSON_PATH = "chargers_locations.geojson" # Use this file

# Population data remains the same
POP_BY_DISTRICT = {
    "Mitte": 397134, "Friedrichshain-Kreuzberg": 293454, "Pankow": 424307,
    "Charlottenburg-Wilmersdorf": 343081, "Spandau": 257091,
    "Steglitz-Zehlendorf": 310446, "Tempelhof-Schöneberg": 355868,
    "Neukölln": 330017, "Treptow-Köpenick": 294081,
    "Marzahn-Hellersdorf": 291948, "Lichtenberg": 311881,
    "Reinickendorf": 268792
}

# BEV Counts from Table I of the Paper
BEV_BY_DISTRICT = {
    "Mitte": 8900, "Friedrichshain-Kreuzberg": 7500, "Pankow": 10500,
    "Charlottenburg-Wilmersdorf": 9800, "Spandau": 4500,
    "Steglitz-Zehlendorf": 8800, "Tempelhof-Schöneberg": 7200,
    "Neukölln": 5900, "Treptow-Köpenick": 5500,
    "Marzahn-Hellersdorf": 4200, "Lichtenberg": 5100,
    "Reinickendorf": 5400
}
N_DISTRICTS = len(POP_BY_DISTRICT)

# --- 1. Helper Functions ---

def canonical_name(name):
    """Maps variant district names from GeoJSON properties to a canonical name."""
    key = str(name).strip()
    # Map known variants observed in the GeoJSON properties
    if "Friedrichshain" in key and "Kreuzberg" in key: return "Friedrichshain-Kreuzberg"
    if "Friedrichshain-Kr" in key: return "Friedrichshain-Kreuzberg"
    if "Mitte" in key: return "Mitte"
    if "Pankow" in key: return "Pankow"
    if "Treptow" in key or "Köpenick" in key: return "Treptow-Köpenick"
    if "Neuk" in key or "Neukölln" in key: return "Neukölln"
    if "Charlottenburg" in key or "Wilmersdorf" in key: return "Charlottenburg-Wilmersdorf"
    if "Spandau" in key: return "Spandau"
    if "Steglitz" in key or "Zehlendorf" in key: return "Steglitz-Zehlendorf"
    if "Tempelhof" in key or "Schöneberg" in key or "Schoeneberg" in key: return "Tempelhof-Schöneberg"
    if "Marzahn" in key or "Hellersdorf" in key: return "Marzahn-Hellersdorf"
    if "Lichtenberg" in key: return "Lichtenberg"
    if "Reinickendorf" in key: return "Reinickendorf"
    # Fallback if no specific match
    return key

def prepare_baseline_data_from_geojson(geojson_file_path):
    """
    Loads GeoJSON data, processes properties, and returns the aggregated
    baseline DataFrame and coordinate dictionary.
    """
    print(f"Preparing baseline data from {geojson_file_path}...")

    # 1. Load GeoJSON
    try:
        gdf = gpd.read_file(geojson_file_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

    except Exception as e:
        print(f"❌ CRITICAL: Could not read {geojson_file_path}. Error: {e}")
        return None, None

    # 2. Filter by date
    # Make cutoff timezone-aware (UTC) as GeoJSON dates likely have 'Z'
    cutoff = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Handle timezone info if present (e.g., 'Z' suffix)
    gdf['inbetriebnahme_dt'] = pd.to_datetime(gdf['inbetriebnahme'], errors='coerce', utc=True)

    # Filter rows where conversion was successful and date is <= cutoff
    gdf_filtered = gdf[gdf['inbetriebnahme_dt'].notna() & (gdf['inbetriebnahme_dt'] <= cutoff)].copy()

    print(f"Found {len(gdf_filtered)} chargers commissioned by {cutoff.date()}.")

    # 3. Aggregate counts and coordinates by canonical district name
    canon_counts = {}
    canon_coords = {}

    for idx, row in gdf_filtered.iterrows():
        # Try multiple potential district name columns
        bezirk_raw = row.get('bezirk') or row.get('bezirk_name') or row.get('district') or "Unknown"
        canon_dist = canonical_name(bezirk_raw)

        if canon_dist not in POP_BY_DISTRICT:
            continue # Skip chargers we can't map

        # Use 'anzahl_ladepunkte', default to 1
        num_chargers = pd.to_numeric(row.get('anzahl_ladepunkte'), errors='coerce')
        if pd.isna(num_chargers) or num_chargers < 1:
            num_chargers = 1
        num_chargers = int(num_chargers)

        canon_counts[canon_dist] = canon_counts.get(canon_dist, 0) + num_chargers

        # Extract coordinates
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        coords = None
        if geom.geom_type == 'Point':
            coords = (geom.x, geom.y) # lon, lat
        elif geom.geom_type == 'MultiPoint':
            if len(geom.geoms) > 0:
                 coords = (geom.geoms[0].x, geom.geoms[0].y)

        if coords:
            canon_coords.setdefault(canon_dist, [])
            for _ in range(num_chargers):
                canon_coords[canon_dist].append(coords)

    # 4. Build final baseline DataFrame
    baseline = []
    berlin_center = (13.4050, 52.5200) # lon, lat fallback

    for b_name in POP_BY_DISTRICT.keys():
        coords_list = canon_coords.get(b_name, [])
        centroid_x = np.mean([c[0] for c in coords_list]) if coords_list else berlin_center[0]
        centroid_y = np.mean([c[1] for c in coords_list]) if coords_list else berlin_center[1]

        baseline.append({
            "district": b_name,
            "existing_chargers": int(canon_counts.get(b_name, 0)),
            "population": POP_BY_DISTRICT[b_name],
            "centroid_x": centroid_x,
            "centroid_y": centroid_y
        })

    df_base = pd.DataFrame(baseline).set_index("district")

    total_pop = df_base["population"].sum()
    df_base["bev_est"] = df_base.index.map(BEV_BY_DISTRICT).fillna(0).astype(int)

    df_base["bev_per_charger_initial"] = df_base["bev_est"] / df_base["existing_chargers"].replace(0, np.nan)
    df_base["bev_per_charger_initial"] = df_base["bev_per_charger_initial"].fillna(df_base["bev_est"])

    print("✅ Baseline data prepared successfully from GeoJSON.")
    # Add check for charger counts
    if df_base["existing_chargers"].sum() == 0:
        print("⚠️ WARNING: No existing chargers found after filtering. Results may be inaccurate.")
    else:
        print(f"Total existing chargers found: {df_base['existing_chargers'].sum()}")
    # print(df_base.head()) # Optional: Display head for verification
    return df_base, canon_coords


def gini(array):
    """Calculates the Gini coefficient."""
    arr = np.array(array, dtype=float)
    if arr.size == 0 or np.all(arr == 0): return 0.0
    arr = arr.flatten()
    if np.any(arr < 0): arr = arr - arr.min()
    n = arr.size
    sorted_arr = np.sort(arr)
    cumvals = np.cumsum(sorted_arr, dtype=float)
    if cumvals[-1] == 0: return 0.0
    g = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return g

def allocate_chargers(df, K, strategy):
    """Applies one of three allocation strategies (A, B, C)."""
    df2 = df.copy()
    df2["new_alloc"] = 0

    if K == 0: # Handle case where K is 0
        df2["post_chargers"] = df2["existing_chargers"]
        df2["post_bev_per_charger"] = df2["bev_per_charger_initial"] # Use initial ratio
        df2["gini_post"] = gini(df2["post_bev_per_charger"].values)
        return df2

    if strategy == "A": # Max Equity
        order = df2.sort_values("bev_per_charger_initial", ascending=False).index.tolist()
    elif strategy == "C": # Max Utility
        order = df2.sort_values("population", ascending=False).index.tolist()
    else: # Strategy B (Balanced)
        base = K // len(df2)
        rem = K % len(df2)
        df2["new_alloc"] = base
        order = df2.sort_values("bev_per_charger_initial", ascending=False).index.tolist()
        for i in range(rem):
            df2.at[order[i % len(order)], "new_alloc"] += 1
        K = 0 # Allocation done

    # Allocate remaining for A and C
    if strategy in ["A", "C"] and K > 0:
        for i in range(K):
            if not order: # Safety check if order list is somehow empty
                break
            target = order[i % len(order)]
            df2.at[target, "new_alloc"] += 1

    df2["post_chargers"] = df2["existing_chargers"] + df2["new_alloc"]
    df2["post_bev_per_charger"] = df2["bev_est"] / df2["post_chargers"].replace(0, np.nan)
    df2["post_bev_per_charger"] = df2["post_bev_per_charger"].fillna(df2["bev_est"])
    df2["gini_post"] = gini(df2["post_bev_per_charger"].values)
    return df2

def run_queue_simulation(df_alloc, freq_per_bev_week=0.2, mean_service_minutes=45, seed=42):
    """Runs a simple M/M/c queuing simulation per district."""
    random.seed(seed)
    np.random.seed(seed)
    period_hours = 7 * 24
    results = {}
    for idx, row in df_alloc.iterrows():
        bev = int(row["bev_est"])
        servers = int(row["post_chargers"])
        if servers <= 0:
            results[idx] = {"avg_wait_min": None, "pct_waited": None, "utilization": 0.0, "n_arrivals": 0}
            continue

        lam = bev * freq_per_bev_week
        n_arrivals = np.random.poisson(lam)
        arrival_times = np.sort(np.random.uniform(0, period_hours, size=n_arrivals))
        service_times = np.random.exponential(scale=(mean_service_minutes / 60.0), size=n_arrivals)

        server_free = [0.0] * servers
        waits = []
        total_service_time = 0.0

        for at, st in zip(arrival_times, service_times):
            idx_server = min(range(servers), key=lambda i: server_free[i])
            free_time = server_free[idx_server]
            wait = max(0.0, free_time - at)
            start = at + wait
            finish = start + st
            server_free[idx_server] = finish
            waits.append(wait * 60.0)
            total_service_time += st

        avg_wait = float(np.mean(waits)) if waits else 0.0
        pct_waited = float(sum(1 for w in waits if w > 1e-6) / len(waits)) if waits else 0.0
        utilization = total_service_time / (servers * period_hours) if servers * period_hours > 0 else 0.0
        results[idx] = {"avg_wait_min": avg_wait, "pct_waited": pct_waited, "utilization": utilization, "n_arrivals": n_arrivals}
    return results

# --- 2. Main Analysis Functions ---

def run_analysis_1_district_sim(df_base):
    """
    Runs the district-level simulation for K=100, 150, 500, 1000.
    """
    print("\n--- Running Analysis 1: District-Level Simulation (A, B, C) ---")
    Ks = [100, 150, 500, 1000]
    strategies = ["A", "B", "C"]
    summary_rows = []
    all_allocations = {}

    for K in Ks:
        for s in strategies:
            df_alloc = allocate_chargers(df_base, K, s)
            sim_res = run_queue_simulation(df_alloc, freq_per_bev_week=0.2, mean_service_minutes=45)

            # Use np.nanmean to handle districts where simulation might not yield results
            valid_waits = [v["avg_wait_min"] for v in sim_res.values() if v["avg_wait_min"] is not None]
            avg_wait = np.mean(valid_waits) if valid_waits else np.nan # Use np.mean

            valid_pct_waited = [v["pct_waited"] for v in sim_res.values() if v["pct_waited"] is not None]
            pct_waited = np.mean(valid_pct_waited) if valid_pct_waited else np.nan # Use np.mean

            valid_util = [v["utilization"] for v in sim_res.values()]
            util = np.mean(valid_util) if valid_util else np.nan # Use np.mean


            pop_served = df_alloc.loc[df_alloc["post_chargers"] > 0, "population"].sum()
            coverage_pct = 100 * pop_served / df_alloc["population"].sum()
            gini_post = df_alloc["gini_post"].iloc[0]

            summary_rows.append({
                "K": K, "Strategy": s, "Gini_post": gini_post, "Coverage_pct": coverage_pct,
                "Avg_wait_min": avg_wait, "Pct_arrivals_waited": pct_waited, "Avg_utilization": util
            })
            all_allocations[(K, s)] = df_alloc.copy()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "analysis1_summary_metrics_geojson.csv", index=False)

    print("Simulation summary (Strategies A, B, C):")
    print(df_summary)
    print("--- Analysis 1 Complete ---")
    return df_summary, all_allocations


def run_analysis_2_sensitivity(df_base, K_focus=100):
    """
    Runs the sensitivity analysis, focused only on K=100.
    """
    print(f"\n--- Running Analysis 2: Sensitivity (K={K_focus}) ---")
    freqs = [0.1, 0.2, 0.3, 0.4]
    services = [30, 45, 60, 90]
    strategies = ["A", "B", "C"]
    summary_records_sens = []

    for strat in strategies:
        df_alloc = allocate_chargers(df_base, K_focus, strat)
        for f in freqs:
            for s in services:
                sim = run_queue_simulation(df_alloc, freq_per_bev_week=f, mean_service_minutes=s, seed=123)

                valid_waits = [v["avg_wait_min"] for v in sim.values() if v["avg_wait_min"] is not None]
                avg_wait = np.mean(valid_waits) if valid_waits else np.nan # Use np.mean

                valid_util = [v["utilization"] for v in sim.values()]
                util = np.mean(valid_util) if valid_util else np.nan # Use np.mean

                summary_records_sens.append({
                    "K": K_focus, "Strategy": strat, "freq": f, "service_min": s,
                    "Avg_wait_min": avg_wait, "Avg_utilization": util
                })

    df_sensitivity = pd.DataFrame(summary_records_sens)
    df_sensitivity.to_csv(OUTPUT_DIR / f"analysis2_sensitivity_summary_K{K_focus}_geojson.csv", index=False)

    print(f"Sensitivity summary for K={K_focus} (sample):")
    print(df_sensitivity.round(3).head())
    print("--- Analysis 2 Complete ---")
    return df_sensitivity


def generate_k100_heatmaps(df_base, canon_coords, all_allocations):
    """
    Generates Folium heatmaps + new markers for K=100 allocations.
    """
    print("\n--- Running Analysis 4: Generating K=100 Heatmaps ---")
    K = 100
    strategies = ["A", "B", "C"]
    berlin_center_latlon = [52.5200, 13.4050] # lat, lon

    existing_locations = []
    for district in canon_coords:
        if district in POP_BY_DISTRICT: # Only include valid districts
            for lon, lat in canon_coords[district]:
                existing_locations.append([lat, lon]) # HeatMap expects [lat, lon]

    if not existing_locations:
        print("⚠️ WARNING: No existing charger locations found in GeoJSON data. Heatmaps will only show new locations.")

    for s in strategies:
        strat_name = {"A": "Max_Equity", "B": "Balanced", "C": "Max_Utility"}[s]

        if (K, s) not in all_allocations:
            print(f"Warning: Allocation for (K={K}, Strategy={s}) not found. Skipping heatmap.")
            continue

        df_alloc = all_allocations[(K, s)]

        new_locations_markers = []
        new_locations_heat = []

        for district, row in df_alloc.iterrows():
            new_chargers_count = int(row["new_alloc"])
            if new_chargers_count > 0:
                # Use centroids calculated during baseline prep
                lon = df_base.at[district, "centroid_x"]
                lat = df_base.at[district, "centroid_y"]

                new_locations_markers.append({
                    "location": [lat, lon],
                    "popup": f"New Allocation: {new_chargers_count}\nDistrict: {district}\nStrategy: {strat_name}"
                })
                for _ in range(new_chargers_count):
                    # Add jitter to heatmap points to avoid perfect overlap at centroid
                    new_locations_heat.append([lat + random.uniform(-0.001, 0.001),
                                               lon + random.uniform(-0.001, 0.001)])


        all_locations_heat = existing_locations + new_locations_heat

        if not all_locations_heat:
             print(f"⚠️ WARNING: No locations (existing or new) found for K={K}, Strat={s}. Skipping heatmap.")
             continue


        m = folium.Map(location=berlin_center_latlon, zoom_start=10, tiles="CartoDB positron")

        HeatMap(all_locations_heat, radius=10, blur=15).add_to(m)

        for marker_info in new_locations_markers:
            folium.Marker(
                location=marker_info["location"],
                popup=marker_info["popup"],
                icon=folium.Icon(color='red', icon='charging-station', prefix='fa')
            ).add_to(m)

        map_filename = OUTPUT_DIR / f"analysis4_heatmap_K{K}_Strat_{strat_name}_geojson.html"
        m.save(str(map_filename))
        print(f"✅ Heatmap for {strat_name} saved to: {map_filename}")

    print("--- Analysis 4 Complete ---")


def generate_nsga2_plot():
    """
    Loads the user-provided 'nsga2_all_k_results.csv' and replicates
    the 'nsga2_pareto_fronts_multi_k.jpg' plot.
    """
    print(f"\n--- Replicating NSGA-II Plot from {NSGA2_FILE_PATH} ---")
    if not os.path.exists(NSGA2_FILE_PATH):
        print(f"❌ Error: {NSGA2_FILE_PATH} not found. Skipping NSGA-II plot.")
        return

    try:
        df_nsga = pd.read_csv(NSGA2_FILE_PATH)
        df_nsga = df_nsga.dropna(subset=['Gini_post', 'Weighted_Coverage_Score'])

        plt.figure(figsize=(10, 7))
        k_values = sorted(df_nsga['K'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

        for k, color in zip(k_values, colors):
            k_data = df_nsga[df_nsga['K'] == k]
            plt.scatter(
                k_data['Gini_post'],
                # Assuming the CSV stores Avg_Wait_Min here as per generator script
                k_data['Weighted_Coverage_Score'],
                color=color, label=f'K={k}', alpha=0.7, s=30
            )

        plt.xlabel("Gini Coefficient (Lower is Better)")
        plt.ylabel("Weighted Population Coverage (Higher is Better)")
        plt.title("NSGA-II Pareto Fronts (from nsga2_all_k_results.csv)")
        plt.legend(title="K Values")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().invert_xaxis() # Gini: Lower is better
        # Do NOT invert Y axis: Higher coverage is better

        plot_filename = OUTPUT_DIR / "nsga2_pareto_fronts_replicated.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ NSGA-II plot saved to: {plot_filename}")
        # plt.show() # Prevent showing plot in this environment
        plt.close() # Close plot figure

    except Exception as e:
        print(f"❌ Error generating NSGA-II plot: {e}")


def generate_avg_times_plot(df_summary):
    """
    Replicates the 'Average times.jpeg' plot using simulation data for K=100.
    """
    print("\n--- Replicating 'Average Times' Plot (K=100, Strategies A,B,C) ---")
    try:
        # Ensure the summary dataframe has the correct columns and K=100 data
        if df_summary is None or 'K' not in df_summary.columns or 'Avg_wait_min' not in df_summary.columns:
             print("❌ Error: Simulation summary data is missing or incomplete. Cannot generate plot.")
             return

        df_k100 = df_summary[df_summary['K'] == 100].copy()

        if df_k100.empty or df_k100['Avg_wait_min'].isnull().all():
            print("⚠️ Warning: No valid K=100 simulation data found. Skipping 'Average Times' plot.")
            return

        df_k100['Strategy_Name'] = df_k100['Strategy'].map({
            'A': 'Max Equity', 'B': 'Balanced', 'C': 'Max Utility'
        })

        plt.figure(figsize=(8, 5))
        barplot = sns.barplot(
            x='Strategy_Name',
            y='Avg_wait_min',
            data=df_k100,
            palette='muted',
            order=['Max Equity', 'Balanced', 'Max Utility']
        )

        plt.title("Average Wait Times for K=100 Allocation (Strategies A, B, C)")
        plt.xlabel("Allocation Strategy")
        plt.ylabel("Average Wait Time (minutes)")

        # Add labels, handle potential NaN values gracefully
        for p in barplot.patches:
             height = p.get_height()
             if pd.notna(height):
                 barplot.annotate(
                     f"{height:.2f} min",
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points'
                 )

        plot_filename = OUTPUT_DIR / "average_wait_times_replicated_K100_geojson.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Average times bar chart saved to: {plot_filename}")
        # plt.show() # Prevent showing plot
        plt.close() # Close plot figure

    except Exception as e:
        print(f"❌ Error generating average times plot: {e}")


# --- 3. Main Execution ---

def main():
    """
    Main function to run the complete analysis pipeline using GeoJSON.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Check for Required Files ---
    if not os.path.exists(CHARGERS_GEOJSON_PATH):
        print("="*50)
        print(f"❌ CRITICAL ERROR: File not found: {CHARGERS_GEOJSON_PATH}")
        print("Please ensure 'chargers_locations.geojson' is uploaded.")
        print("="*50)
        return

    if not os.path.exists(NSGA2_FILE_PATH):
        print("="*50)
        print(f"⚠️ WARNING: File not found: {NSGA2_FILE_PATH}")
        print("The NSGA-II Pareto front plot will not be generated.")
        print("="*50)

    # --- Step 2: Prepare Baseline Data from GeoJSON ---
    df_base, canon_coords = prepare_baseline_data_from_geojson(CHARGERS_GEOJSON_PATH)
    if df_base is None:
        print("Halting script due to data preparation failure.")
        return

    # --- Step 3: Run Analyses (A, B, C) ---
    df_summary, all_allocations = run_analysis_1_district_sim(df_base)
    run_analysis_2_sensitivity(df_base, K_focus=100)

    # --- Step 4: Generate Required Outputs ---
    generate_k100_heatmaps(df_base, canon_coords, all_allocations)
    generate_avg_times_plot(df_summary)
    generate_nsga2_plot() # This reads the static CSV

    print("\n==================================")
    print("All analyses complete.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")
    print("==================================")

if __name__ == "__main__":
    # Set plot style
    sns.set_theme(style="whitegrid")

    # Run the main pipeline
    main()
