# geo-spatial-Berlin-EVC-placement
This is for collaboration and sharing the code for a paper working on geo-spatial data from Berlin EV center placement

# Equitable EV Charger Deployment Optimization in Berlin

This project replicates the multi-objective geospatial optimization framework detailed in the paper, using the NSGA-II algorithm to balance coverage and spatial equity in EV charger deployment across Berlin's districts.

## Project Structure

- `evc_optimization.py`: The main Python script containing all data processing, NSGA-II logic, simulation, and output generation.
- `requirements.txt`: Lists all necessary Python dependencies.
- `replication_results_nsga2/`: Directory where all generated figures (PNG) and data (CSV) will be saved.

## Setup and Execution

### 1. Install Dependencies

Ensure you have Python installed (3.8+ recommended). Install the required libraries using pip:

```bash
pip install -r requirements.txt
