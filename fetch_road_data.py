"""
Fetch road classification data from OpenStreetMap for METR-LA sensors.

Queries the Overpass API in small batches to find the nearest road segment to each
sensor and extracts: highway tag, lanes, maxspeed, and name.

Output: data/METR-LA/road_features.npz and data/METR-LA/road_features.csv
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pickle

SENSOR_LOCATIONS_CSV = 'data/sensor_graph/graph_sensor_locations.csv'
ADJ_PKL = 'data/sensor_graph/adj_mx.pkl'
OUTPUT_NPZ = 'data/METR-LA/road_features.npz'
OUTPUT_CSV = 'data/METR-LA/road_features.csv'
CACHE_DIR = 'data/road_cache'

OVERPASS_URL = 'https://overpass.kumi.systems/api/interpreter'
SEARCH_RADIUS = 50
BATCH_SIZE = 10

HIGHWAY_CATEGORIES = {
    'motorway': 0,
    'motorway_link': 1,
    'trunk': 2,
    'trunk_link': 3,
    'primary': 4,
    'primary_link': 5,
    'secondary': 6,
    'secondary_link': 7,
    'tertiary': 8,
    'tertiary_link': 9,
    'residential': 10,
    'living_street': 11,
    'service': 12,
    'unclassified': 13,
    'road': 14,
}


def load_sensor_info():
    with open(ADJ_PKL, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    df = pd.read_csv(SENSOR_LOCATIONS_CSV)
    id_to_latlon = {
        str(int(row['sensor_id'])): (row['latitude'], row['longitude'])
        for _, row in df.iterrows()
    }
    locations = []
    for sid in sensor_ids:
        lat, lon = id_to_latlon[str(sid)]
        locations.append((float(lat), float(lon)))
    return sensor_ids, locations


def query_overpass_batch(sensor_indices, locations, radius=SEARCH_RADIUS):
    import urllib.request
    import urllib.parse

    parts = []
    for i in sensor_indices:
        lat, lon = locations[i]
        parts.append(f'way["highway"](around:{radius},{lat},{lon});')
    query = '[out:json][timeout:60];\n(' + '\n'.join(parts) + '\n);\nout body;>;out skel qt;'
    post_data = urllib.parse.urlencode({'data': query}).encode('utf-8')
    req = urllib.request.Request(OVERPASS_URL, data=post_data)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.loads(response.read().decode())
            return result
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    return None


def find_nearest_for_sensor(elements_by_batch, batch_nodes, lat, lon):
    best_dist_sq = float('inf')
    best_tags = None
    for nodes, ways in elements_by_batch:
        for way_coords, tags in ways:
            for nlat, nlon in way_coords:
                d = (nlat - lat) ** 2 + (nlon - lon) ** 2
                if d < best_dist_sq:
                    best_dist_sq = d
                    best_tags = tags

    if best_tags is None:
        return {'highway': 'unknown', 'lanes': 0, 'maxspeed': '', 'name': ''}

    lanes_str = best_tags.get('lanes', '0')
    try:
        lanes = int(float(lanes_str))
    except (ValueError, TypeError):
        lanes = 0

    return {
        'highway': best_tags.get('highway', 'unknown'),
        'lanes': lanes,
        'maxspeed': best_tags.get('maxspeed', ''),
        'name': best_tags.get('name', ''),
    }


def parse_overpass_elements(overpass_result):
    nodes = {}
    for elem in overpass_result.get('elements', []):
        if elem['type'] == 'node':
            nodes[elem['id']] = (elem['lat'], elem['lon'])

    ways = []
    for elem in overpass_result.get('elements', []):
        if elem['type'] == 'way' and 'tags' in elem and 'highway' in elem['tags']:
            way_node_ids = elem.get('nodes', [])
            way_coords = [(nodes[n][0], nodes[n][1]) for n in way_node_ids if n in nodes]
            if way_coords:
                ways.append((way_coords, elem['tags']))

    return nodes, ways


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    sensor_ids, locations = load_sensor_info()
    num_sensors = len(sensor_ids)
    print(f"Fetching road data for {num_sensors} sensors in batches of {BATCH_SIZE}...")

    all_road_info = [None] * num_sensors
    all_batch_elements = []

    for batch_start in range(0, num_sensors, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_sensors)
        batch_indices = list(range(batch_start, batch_end))
        batch_key = f"batch_{batch_start:03d}_{batch_end:03d}"
        cache_file = os.path.join(CACHE_DIR, f'{batch_key}.json')

        if os.path.exists(cache_file):
            print(f"  [{batch_start+1}-{batch_end}/{num_sensors}] Loading cached {batch_key}")
            with open(cache_file, 'r') as f:
                overpass_result = json.load(f)
        else:
            print(f"  [{batch_start+1}-{batch_end}/{num_sensors}] Querying Overpass for sensors {batch_start+1}-{batch_end}...")
            overpass_result = query_overpass_batch(batch_indices, locations)
            if overpass_result is None:
                print(f"    FAILED - skipping batch")
                continue
            with open(cache_file, 'w') as f:
                json.dump(overpass_result, f)
            time.sleep(2)

        nodes, ways = parse_overpass_elements(overpass_result)
        print(f"    Got {len(nodes)} nodes, {len(ways)} roads")
        all_batch_elements.append((nodes, ways))

        for i in batch_indices:
            lat, lon = locations[i]
            all_road_info[i] = find_nearest_for_sensor(
                [(nodes, ways)], None, lat, lon
            )

    # For sensors that didn't get matched in their batch, try across all batches
    for i in range(num_sensors):
        if all_road_info[i] is None or all_road_info[i]['highway'] == 'unknown':
            lat, lon = locations[i]
            all_road_info[i] = find_nearest_for_sensor(all_batch_elements, None, lat, lon)

    # Fill remaining unknowns
    for i in range(num_sensors):
        if all_road_info[i] is None:
            all_road_info[i] = {'highway': 'unknown', 'lanes': 0, 'maxspeed': '', 'name': ''}

    highway_types = [r['highway'] for r in all_road_info]
    lanes_list = [r['lanes'] for r in all_road_info]
    maxspeed_list = [r.get('maxspeed', '') for r in all_road_info]
    name_list = [r.get('name', '') for r in all_road_info]

    highway_encoded = np.array([
        HIGHWAY_CATEGORIES.get(h, len(HIGHWAY_CATEGORIES))
        for h in highway_types
    ], dtype=np.int32)
    lanes_arr = np.array(lanes_list, dtype=np.int32)
    sensor_ids_arr = np.array([int(s) for s in sensor_ids])

    df = pd.DataFrame({
        'sensor_id': [int(s) for s in sensor_ids],
        'latitude': [loc[0] for loc in locations],
        'longitude': [loc[1] for loc in locations],
        'highway': highway_types,
        'highway_encoded': highway_encoded,
        'lanes': lanes_list,
        'maxspeed': maxspeed_list,
        'road_name': name_list,
    })
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nRoad type distribution:")
    print(df['highway'].value_counts().to_string())
    print(f"\nLanes distribution:")
    print(df['lanes'].value_counts().sort_index().to_string())
    print(f"\nLanes=0 (unknown): {(df['lanes'] == 0).sum()}")
    print(f"Unknown highway: {(df['highway'] == 'unknown').sum()}")

    np.savez_compressed(
        OUTPUT_NPZ,
        highway_type=highway_encoded,
        lanes=lanes_arr,
        highway_raw=np.array(highway_types),
        sensor_ids=sensor_ids_arr,
    )
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_NPZ}")


if __name__ == '__main__':
    main()