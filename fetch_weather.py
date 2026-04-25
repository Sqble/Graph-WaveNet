"""
Fetch historical weather data from Open-Meteo for METR-LA sensors.

Output: data/METR-LA/weather.npz
  Shape: (34272, 207, 3)
  Features: [temperature_2m, precipitation, relative_humidity_2m]
  Timestep: 5-minute intervals (interpolated from hourly)
"""

import os
import json
import time
import urllib.request
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

SENSOR_LOCATIONS_CSV = 'data/sensor_graph/graph_sensor_locations.csv'
OUTPUT_PATH = 'data/METR-LA/weather.npz'
CACHE_DIR = 'data/weather_cache'

# Match METR-LA time range
START_DATE = '2012-03-01'
END_DATE = '2012-06-30'

# Weather variables to fetch
VARIABLES = 'temperature_2m,precipitation,relative_humidity_2m'

# METR-LA has 34272 timesteps at 5-minute intervals
NUM_TIMESTEPS = 34272
NUM_SENSORS = 207


def load_sensor_locations():
    """Load sensor IDs, latitudes, longitudes in METR-LA order."""
    df = pd.read_csv(SENSOR_LOCATIONS_CSV)
    # METR-LA sensor order comes from adj_mx.pkl
    import pickle
    sensor_ids, _, _ = pickle.load(
        open('data/sensor_graph/adj_mx.pkl', 'rb'), encoding='latin1'
    )
    # Build lookup (convert both to strings for matching)
    id_to_latlon = {
        str(int(row['sensor_id'])): (row['latitude'], row['longitude'])
        for _, row in df.iterrows()
    }
    locations = []
    for sid in sensor_ids:
        lat, lon = id_to_latlon[str(sid)]
        locations.append((float(lat), float(lon)))
    return locations


def fetch_weather(lat, lon, start_date, end_date, variables):
    """Fetch hourly weather from Open-Meteo archive API."""
    url = (
        f'https://archive-api.open-meteo.com/v1/archive?'
        f'latitude={lat}&longitude={lon}'
        f'&start_date={start_date}&end_date={end_date}'
        f'&hourly={variables}'
        f'&timezone=America%2FLos_Angeles'
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.loads(response.read().decode())
    return data['hourly']


def interpolate_hourly_to_5min(hourly_data, num_timesteps):
    """
    Interpolate hourly weather data to 5-minute intervals.
    
    hourly_data: dict with keys 'time', 'temperature_2m', etc.
    Returns: np.ndarray of shape (num_timesteps, num_features)
    """
    # Parse timestamps
    times = [datetime.fromisoformat(t) for t in hourly_data['time']]
    # Convert to hours since start
    hours = np.array([(t - times[0]).total_seconds() / 3600.0 for t in times])
    
    # Target: 5-minute intervals, num_timesteps total
    # Each timestep is 5/60 = 1/12 hour
    target_hours = np.arange(num_timesteps) / 12.0
    
    features = []
    for key in hourly_data:
        if key == 'time':
            continue
        values = np.array(hourly_data[key], dtype=np.float32)
        # Linear interpolation
        f = interp1d(hours, values, kind='linear', fill_value='extrapolate')
        interpolated = f(target_hours)
        features.append(interpolated)
    
    return np.stack(features, axis=1)  # (num_timesteps, num_features)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("Loading sensor locations...")
    locations = load_sensor_locations()
    assert len(locations) == NUM_SENSORS, f"Expected {NUM_SENSORS} sensors, got {len(locations)}"
    
    # Result array: (timesteps, sensors, features)
    weather = np.zeros((NUM_TIMESTEPS, NUM_SENSORS, 3), dtype=np.float32)
    feature_names = ['temperature_2m', 'precipitation', 'relative_humidity_2m']
    
    for i, (lat, lon) in enumerate(locations):
        cache_file = os.path.join(CACHE_DIR, f'sensor_{i:03d}_{lat:.5f}_{lon:.5f}.json')
        
        if os.path.exists(cache_file):
            print(f"  [{i+1}/{NUM_SENSORS}] Loading cached data for sensor {i}")
            with open(cache_file, 'r') as f:
                hourly_data = json.load(f)
        else:
            print(f"  [{i+1}/{NUM_SENSORS}] Fetching weather for sensor {i} ({lat:.4f}, {lon:.4f})")
            try:
                hourly_data = fetch_weather(lat, lon, START_DATE, END_DATE, VARIABLES)
                with open(cache_file, 'w') as f:
                    json.dump(hourly_data, f)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            # Rate limit: sleep 0.3s between requests (~200 req/min)
            time.sleep(0.3)
        
        # Interpolate to 5-minute intervals
        sensor_weather = interpolate_hourly_to_5min(hourly_data, NUM_TIMESTEPS)
        weather[:, i, :] = sensor_weather
    
    # Save
    np.savez_compressed(OUTPUT_PATH, data=weather)
    print(f"\nSaved weather data to {OUTPUT_PATH}")
    print(f"Shape: {weather.shape} (timesteps={NUM_TIMESTEPS}, sensors={NUM_SENSORS}, features={len(feature_names)})")
    print(f"Features: {feature_names}")
    
    # Print some stats
    print("\nWeather data statistics:")
    for j, name in enumerate(feature_names):
        vals = weather[:, :, j]
        print(f"  {name}: mean={vals.mean():.2f}, std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")


if __name__ == '__main__':
    main()
