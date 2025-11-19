#!/usr/bin/env python

"""
SCRIPT 1: NSGA-II GENERATOR (Run this on your local computer)

This script generates the NSGA-II optimization results for the
Berlin EV charger placement problem.

It MUST be run locally after installing 'pymoo'.
"""

import os
import json
import math
import random
import glob
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import geopandas as gpd
import requests

# --- Prerequisite: pymoo library ---
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
except ImportError:
    print("="*50)
    print("CRITICAL ERROR: 'pymoo' library not found.")
    print("Please install it on your local machine: pip install pymoo pandas geopandas requests")
    print("This script cannot run without it.")
    print("="*50)
    sys.exit(1) # Force script to stop

# --- 1. Baseline Data Functions ---

OUTPUT_DIR = Path("./simulation_outputs_k100")
OUTPUT_DIR = Path("./simulation_outputs_k100")
CHARGERS_GEOJSON_PATH = "chargers_locations.geojson"

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

def canonical_name(name):
    """Maps variant district names from GeoJSON properties to a canonical name."""
    key = str(name).strip()
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
    return key

def prepare_baseline_data_from_geojson(geojson_file_path):
    print(f"Preparing baseline data from {geojson_file_path}...")
    try:
        gdf = gpd.read_file(geojson_file_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
    except Exception as e:
        print(f"❌ CRITICAL: Could not read {geojson_file_path}. Error: {e}")
        return None

    # Filter by date (cutoff 2024-12-31)
    cutoff = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    gdf['inbetriebnahme_dt'] = pd.to_datetime(gdf['inbetriebnahme'], errors='coerce', utc=True)
    gdf_filtered = gdf[gdf['inbetriebnahme_dt'].notna() & (gdf['inbetriebnahme_dt'] <= cutoff)].copy()
    
    print(f"Found {len(gdf_filtered)} chargers commissioned by {cutoff.date()}.")

    canon_counts = {}
    for idx, row in gdf_filtered.iterrows():
        bezirk_raw = row.get('bezirk') or row.get('bezirk_name') or row.get('district') or "Unknown"
        canon_dist = canonical_name(bezirk_raw)
        if canon_dist not in POP_BY_DISTRICT: continue
        
        num_chargers = pd.to_numeric(row.get('anzahl_ladepunkte'), errors='coerce')
        if pd.isna(num_chargers) or num_chargers < 1: num_chargers = 1
        canon_counts[canon_dist] = canon_counts.get(canon_dist, 0) + int(num_chargers)

    baseline = []
    for b_name in POP_BY_DISTRICT.keys():
        baseline.append({
            "district": b_name,
            "existing_chargers": int(canon_counts.get(b_name, 0)),
            "population": POP_BY_DISTRICT[b_name],
        })
    df_base = pd.DataFrame(baseline).set_index("district")
    
    df_base["bev_est"] = df_base.index.map(BEV_BY_DISTRICT).fillna(0).astype(int)
    df_base["bev_per_charger_initial"] = df_base["bev_est"] / df_base["existing_chargers"].replace(0, np.nan)
    df_base["bev_per_charger_initial"] = df_base["bev_per_charger_initial"].fillna(df_base["bev_est"])

    print("✅ Baseline data prepared successfully from GeoJSON.")
    return df_base

# --- 2. Simulation Functions ---

def gini(array):
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

# Simulation function removed from optimization loop to align with paper methodology.
# Simulation will be run separately in the analysis/visualization phase.

# --- 3. NSGA-II Optimization Problem Definition ---

class ChargerAllocationProblem(Problem):
    """
    Defines the multi-objective optimization problem for pymoo.
    """
    def __init__(self, df_base, K):
        self.df_base = df_base
        self.K = K

        super().__init__(
            n_var=N_DISTRICTS,
            n_obj=2,
            n_constr=1,
            xl=0,
            xu=K,
            elementwise=True,
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Constraint: sum of allocated chargers must equal K
        g1 = (np.sum(x) - self.K) ** 2
        out["G"] = [g1]

        # --- Objectives ---
        df_alloc = self.df_base.copy()
        df_alloc["new_alloc"] = x
        df_alloc["post_chargers"] = df_alloc["existing_chargers"] + df_alloc["new_alloc"]

        df_alloc["post_bev_per_charger"] = df_alloc["bev_est"] / df_alloc["post_chargers"].replace(0, np.nan)
        df_alloc["post_bev_per_charger"] = df_alloc["post_bev_per_charger"].fillna(df_alloc["bev_est"])

        # Objective 1: Minimize Gini (Equity)
        f1 = gini(df_alloc["post_bev_per_charger"].values)

        # Objective 2: Maximize Weighted Population Coverage (Utility)
        # We maximize the sum of (new_chargers * population_of_district)
        # Since pymoo minimizes, we minimize the negative sum.
        # This aligns with "placing chargers in areas with high population density".
        
        # Note: The paper mentions "pop_coverage(j)" for specific locations. 
        # As we are optimizing district counts, we use population weighting as the proxy.
        weighted_coverage = np.sum(x * df_alloc["population"].values)
        f2 = -1 * weighted_coverage

        out["F"] = [f1, f2]

# --- 4. Main Execution ---

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load baseline data
    if not os.path.exists(CHARGERS_GEOJSON_PATH):
        print(f"❌ CRITICAL: {CHARGERS_GEOJSON_PATH} not found. Exiting.")
        return

    df_base = prepare_baseline_data_from_geojson(CHARGERS_GEOJSON_PATH)
    if df_base is None:
        print("❌ CRITICAL: Data preparation failed. Exiting.")
        return

    # 2. Define K values to run
    K_values = [25, 50, 100, 150]
    all_results = []

    for K in K_values:
        print(f"\n--- Running NSGA-II Optimization for K = {K} ---")

        problem = ChargerAllocationProblem(df_base, K)

        algorithm = NSGA2(
            pop_size=100,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=int, repair=RoundingRepair()),
            mutation=PM(prob=0.05, eta=20, vtype=int, repair=RoundingRepair()), # Mutation rate 0.05 as per paper
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', 200),
            verbose=True,
            seed=42
        )

        print(f"--- Optimization for K = {K} Complete ---")

        if res.X is not None:
            for i in range(len(res.X)):
                solution_vars = res.X[i]
                solution_objs = res.F[i]

                # We store Avg_Wait_Min in the 'Weighted_Coverage_Score' column
                # to match the format of the file you provided.
                all_results.append({
                    "Replication": 0,
                    "K": K,
                    "Gini_post": solution_objs[0],
                    "Weighted_Coverage_Score": -1 * solution_objs[1], # Convert back to positive for reporting
                    "Allocation": str(list(solution_vars))
                })
        else:
            print(f"No valid solutions found for K = {K}.")

    # 7. Save the final DataFrame to CSV
    df_results = pd.DataFrame(all_results)

    output_filename = "nsga2_all_k_results_GENERATED.csv"
    df_results.to_csv(output_filename, index=False)

    print("\n=======================================================")
    print(f"✅ All optimizations complete!")
    print(f"Results saved to: {output_filename}")
    print("Column 'Weighted_Coverage_Score' contains the weighted population coverage (Utility).")
    print("=======================================================")

if __name__ == "__main__":
    main()
