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
from datetime import datetime

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
DISTRICTS_GEOJSON_PATH = "berlin_districts.geojson"
DISTRICTS_DOWNLOAD_URL = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"

POP_BY_DISTRICT = {
    "Mitte": 397134, "Friedrichshain-Kreuzberg": 293454, "Pankow": 424307,
    "Charlottenburg-Wilmersdorf": 343081, "Spandau": 257091,
    "Steglitz-Zehlendorf": 310446, "Tempelhof-Schöneberg": 355868,
    "Neukölln": 330017, "Treptow-Köpenick": 294081,
    "Marzahn-Hellersdorf": 291948, "Lichtenberg": 311881,
    "Reinickendorf": 268792
}
N_DISTRICTS = len(POP_BY_DISTRICT)

def find_latest_bnetza_file():
    files = glob.glob("Ladesaeulenregister_BNetzA_*.csv")
    if not files: return None
    files.sort()
    print(f"Using data file: {files[-1]}")
    return files[-1]

def download_berlin_districts():
    if os.path.exists(DISTRICTS_GEOJSON_PATH):
        print(f"Using existing district file: {DISTRICTS_GEOJSON_PATH}")
        return True
    print(f"Downloading Berlin district boundaries from {DISTRICTS_DOWNLOAD_URL}...")
    try:
        response = requests.get(DISTRICTS_DOWNLOAD_URL, timeout=30)
        response.raise_for_status()
        with open(DISTRICTS_GEOJSON_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✅ District boundaries saved as {DISTRICTS_GEOJSON_PATH}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"District download failed: {e}")
        return False

def find_col(df_columns, possibilities):
    """Finds the correct column name from a list of possibilities."""
    for col in possibilities:
        if col in df_columns:
            return col
    # If not found, try lowercase versions robustly
    possibilities_lower = {p.lower() for p in possibilities}
    for col in df_columns:
        if col.lower() in possibilities_lower:
            return col
    return None


def prepare_baseline_data_from_bneza(bnetza_file_path):
    try:
        gdf_districts = gpd.read_file(DISTRICTS_GEOJSON_PATH).to_crs(epsg=4326)
    except Exception as e:
        print(f"❌ CRITICAL: Could not read {DISTRICTS_GEOJSON_PATH}. Error: {e}")
        return None

    DISTRICT_COL = 'name' # Correct column in the GeoJSON
    if DISTRICT_COL not in gdf_districts.columns:
        print(f"❌ CRITICAL: Could not find district name column '{DISTRICT_COL}' in {DISTRICTS_GEOJSON_PATH}.")
        print(f"Found columns: {gdf_districts.columns.to_list()}")
        return None

    try:
        # *** FINAL FIX: Changed header=10 to header=11 ***
        df_bnetza = pd.read_csv(
            bnetza_file_path,
            sep=';',
            encoding='latin-1',
            decimal=',',
            header=11,  # Skip first 11 rows, read headers from row 12 (0-indexed)
            low_memory=False
        )
    except Exception as e:
        print(f"❌ CRITICAL: Failed to read BNetzA CSV file (using header=11, latin-1). Error: {e}")
        return None

    # --- Robust Column Finding ---
    all_cols = df_bnetza.columns
    # Clean up column names (remove leading/trailing spaces if any)
    df_bnetza.columns = df_bnetza.columns.str.strip()
    all_cols = df_bnetza.columns # Update after stripping

    city_col = find_col(all_cols, ['Ort'])
    date_col = find_col(all_cols, ['Inbetriebnahmedatum'])
    lat_col = find_col(all_cols, ['Breitengrad'])
    lon_col = find_col(all_cols, ['Längengrad'])
    count_col = find_col(all_cols, ['Anzahl Ladepunkte'])

    if not all([city_col, date_col, lat_col, lon_col, count_col]):
        print("❌ CRITICAL: Could not find one or more required columns AFTER skipping header rows (header=11).")
        if not city_col: print("   - Missing: 'Ort' (City)")
        if not date_col: print("   - Missing: 'Inbetriebnahmedatum' (Date)")
        if not lat_col: print("   - Missing: 'Breitengrad' (Latitude)")
        if not lon_col: print("   - Missing: 'Längengrad' (Longitude)")
        if not count_col: print("   - Missing: 'Anzahl Ladepunkte' (Count)")
        print(f"Found columns in row 12: {all_cols.to_list()}") # Show columns found after skipping
        return None

    print(f"Using columns: City='{city_col}', Date='{date_col}', Lat='{lat_col}', Lon='{lon_col}', Count='{count_col}'")

    df_berlin = df_bnetza[df_bnetza[city_col] == 'Berlin'].copy()

    df_berlin[date_col] = pd.to_datetime(
        df_berlin[date_col], dayfirst=True, errors='coerce'
    )
    df_berlin = df_berlin.dropna(subset=[date_col])
    cutoff = datetime(2024, 12, 31)
    df_berlin_filtered = df_berlin[df_berlin[date_col] <= cutoff].copy()

    df_berlin_filtered[lat_col] = pd.to_numeric(df_berlin_filtered[lat_col], errors='coerce')
    df_berlin_filtered[lon_col] = pd.to_numeric(df_berlin_filtered[lon_col], errors='coerce')
    df_berlin_filtered = df_berlin_filtered.dropna(subset=[lat_col, lon_col])

    gdf_points = gpd.GeoDataFrame(
        df_berlin_filtered,
        geometry=gpd.points_from_xy(df_berlin_filtered[lon_col], df_berlin_filtered[lat_col]),
        crs="EPSG:4326"
    )

    gdf_joined = gpd.sjoin(gdf_points, gdf_districts, how='left', predicate='within')

    # Group by the correct district and count columns
    canon_counts = gdf_joined.groupby(DISTRICT_COL)[count_col].sum().to_dict()

    baseline = []
    for b_name in POP_BY_DISTRICT.keys():
        baseline.append({
            "district": b_name,
            "existing_chargers": int(canon_counts.get(b_name, 0)),
            "population": POP_BY_DISTRICT[b_name],
        })
    df_base = pd.DataFrame(baseline).set_index("district")
    total_pop = df_base["population"].sum()
    df_base["bev_est"] = (50802 * df_base["population"] / total_pop).round().astype(int)
    df_base["bev_per_charger_initial"] = df_base["bev_est"] / df_base["existing_chargers"].replace(0, np.nan)
    df_base["bev_per_charger_initial"] = df_base["bev_per_charger_initial"].fillna(df_base["bev_est"])

    print("✅ Baseline data prepared successfully.")
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

def run_queue_simulation(df_alloc, freq_per_bev_week=0.2, mean_service_minutes=45, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    period_hours = 7 * 24
    results = {}
    for idx, row in df_alloc.iterrows():
        bev = int(row["bev_est"])
        servers = int(row["post_chargers"])
        if servers <= 0:
            results[idx] = {"avg_wait_min": None}
            continue

        lam = bev * freq_per_bev_week
        n_arrivals = np.random.poisson(lam)
        arrival_times = np.sort(np.random.uniform(0, period_hours, size=n_arrivals))
        service_times = np.random.exponential(scale=(mean_service_minutes / 60.0), size=n_arrivals)

        server_free = [0.0] * servers
        waits = []
        for at, st in zip(arrival_times, service_times):
            idx_server = min(range(servers), key=lambda i: server_free[i])
            free_time = server_free[idx_server]
            wait = max(0.0, free_time - at)
            start = at + wait
            finish = start + st
            server_free[idx_server] = finish
            waits.append(wait * 60.0)

        avg_wait = float(np.mean(waits)) if waits else 0.0
        results[idx] = {"avg_wait_min": avg_wait}
    return results

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

        # Objective 1: Minimize Gini
        f1 = gini(df_alloc["post_bev_per_charger"].values)

        # Objective 2: Minimize Average Wait Time
        sim_res = run_queue_simulation(df_alloc)
        all_waits = [v["avg_wait_min"] for v in sim_res.values() if v["avg_wait_min"] is not None]

        if not all_waits:
             f2 = 1e6 # Assign a large penalty if simulation fails
        else:
             f2 = np.mean(all_waits)

        out["F"] = [f1, f2]

# --- 4. Main Execution ---

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load baseline data
    bnetza_file = find_latest_bnetza_file()
    if not bnetza_file:
        print("❌ CRITICAL: No BNetzA file found. Exiting.")
        return
    if not download_berlin_districts():
        print("❌ CRITICAL: Could not download district boundaries. Exiting.")
        return

    df_base = prepare_baseline_data_from_bneza(bnetza_file)
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
            mutation=PM(prob=0.1, eta=20, vtype=int, repair=RoundingRepair()),
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
                    "Weighted_Coverage_Score": solution_objs[1],
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
    print("Column 'Weighted_Coverage_Score' contains 'Avg_wait_min'.")
    print("=======================================================")

if __name__ == "__main__":
    main()
