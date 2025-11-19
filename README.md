# Berlin EV Charger Placement Simulation

This project provides tools to simulate and visualize the optimal placement of Electric Vehicle (EV) chargers in Berlin using NSGA-II optimization.

## Project Structure

- `download_data.py`: Downloads the latest charger location data from Berlin's WFS service.
- `generate_results.py`: Runs the NSGA-II optimization to determine optimal charger allocations.
- `visualize.py`: Generates heatmaps and plots based on the simulation results.
- `requirements.txt`: List of Python dependencies.

## Setup

1.  **Install Dependencies**:
    Ensure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download Data
Download the latest charger locations:
```bash
python download_data.py
```
This will create `chargers_locations.geojson`.

### 2. Run Simulation (Optional)
> **Note**: This step requires the raw BNetzA CSV file (e.g., `Ladesaeulenregister_BNetzA_*.csv`) to be present in the directory.

Run the optimization algorithm:
```bash
python generate_results.py
```
This will generate `nsga2_all_k_results_GENERATED.csv`.

### 3. Visualize Results
Generate charts and heatmaps using the existing results:
```bash
python visualize.py
```
This will create a `simulation_outputs_final_geojson` directory containing:
- Heatmaps (`.html`)
- Summary metrics (`.csv`)
- Plots (`.png`)

## Data Sources
- **Charger Locations**: [Berlin Open Data (WFS)](https://gdi.berlin.de/services/wfs/eladesaeulen)
- **District Boundaries**: [Berlin Geodaten](https://github.com/funkeinteraktiv/Berlin-Geodaten)
