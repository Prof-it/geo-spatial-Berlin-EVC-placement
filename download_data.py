import requests
import geopandas as gpd
import tempfile
import os

def download_charger_data():
    # The base WFS URL from the capabilities document
    wfs_url = "https://gdi.berlin.de/services/wfs/eladesaeulen"

    # CORRECTED: Use a valid typename from the WFS capabilities document
    # This layer provides granular location data for individual charging stations
    params = {
        'service': 'WFS',
        'version': '1.1.0',
        'request': 'GetFeature',
        'typename': 'eladesaeulen:lades_standorte', # Corrected typename
        'outputFormat': 'json',
        'srsName': 'EPSG:4326'
    }

    print("Attempting to download granular charger data using a valid typename...")
    try:
        response = requests.get(wfs_url, params=params, timeout=30)
        response.raise_for_status() # Raise an error for bad status codes

        # Save the response to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".geojson") as tmp_file:
            tmp_file.write(response.text)
            tmp_file_path = tmp_file.name

        try:
            # Read from the temporary file using geopandas
            data = gpd.read_file(tmp_file_path)
            output_file = "chargers_locations.geojson"
            data.to_file(output_file, driver='GeoJSON')
            print(f"✅ Charger location data successfully downloaded and saved as {output_file}")
            print("You can now proceed to the next steps of the simulation.")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred during the download. Please check your internet connection and try again.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"❌ An error occurred while processing the downloaded data: {e}")

if __name__ == "__main__":
    download_charger_data()
