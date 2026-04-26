"""
Visualize road feature data for METR-LA sensor network.

Usage:
    python visualize_road.py

Outputs saved to figures/:
    - road_spatial_map.png          Sensor map colored by road type
    - road_lanes_map.png           Sensor map colored by lane count
    - road_type_distribution.png   Bar chart of road categories
    - road_lanes_distribution.png  Histogram of lane counts
    - road_speed_correlation.png   Box plots of speed vs road type
    - road_network_graph.png       Adjacency matrix + road type similarity
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURE_DIR = 'figures'
ROAD_CSV = 'data/METR-LA/road_features.csv'
ADJ_PKL = 'data/sensor_graph/adj_mx.pkl'
TRAFFIC_H5 = 'data/metr-la.h5'

os.makedirs(FIGURE_DIR, exist_ok=True)

CATEGORY_COLORS = {
    'motorway': '#e6194B',
    'motorway_link': '#f58676',
    'trunk': '#3cb44b',
    'trunk_link': '#7ece8b',
    'primary': '#4363d8',
    'primary_link': '#7a97ed',
    'secondary': '#f58231',
    'secondary_link': '#f9ab6d',
    'tertiary': '#911eb4',
    'tertiary_link': '#b763cc',
    'residential': '#42d4f4',
    'living_street': '#a0e8f0',
    'service': '#bfef45',
    'unclassified': '#f032e6',
    'road': '#fabed4',
    'path': '#ffe119',
    'footway': '#808080',
    'steps': '#cccccc',
    'unknown': '#aaaaaa',
}

ROAD_TYPE_SHORT = {
    'motorway': 'Freeway',
    'motorway_link': 'Freeway Ramp',
    'trunk': 'Major Arterial',
    'trunk_link': 'Arterial Ramp',
    'primary': 'Primary Road',
    'primary_link': 'Primary Ramp',
    'secondary': 'Secondary Road',
    'secondary_link': 'Secondary Ramp',
    'tertiary': 'Tertiary Road',
    'tertiary_link': 'Tertiary Ramp',
    'residential': 'Residential',
    'living_street': 'Living Street',
    'service': 'Service Road',
    'unclassified': 'Unclassified',
    'road': 'Road (Generic)',
    'path': 'Path',
    'footway': 'Footway',
    'steps': 'Steps',
    'unknown': 'Unknown',
}

print("Loading data...")
df = pd.read_csv(ROAD_CSV)
print(f"Loaded {len(df)} sensors from {ROAD_CSV}")

_, _, adj_mx = pickle.load(open(ADJ_PKL, 'rb'), encoding='latin1')

traffic_df = pd.read_hdf(TRAFFIC_H5, key='/df')
avg_speed = traffic_df.mean(axis=0).values
std_speed = traffic_df.std(axis=0).values

# ============================================================
# 1. Spatial Map Colored by Road Type
# ============================================================
print("Plotting spatial map by road type...")
fig, ax = plt.subplots(figsize=(12, 10))

unique_types = df['highway'].unique().tolist()
legend_handles = []
for hwy_type in sorted(unique_types):
    mask = df['highway'] == hwy_type
    color = CATEGORY_COLORS.get(hwy_type, '#cccccc')
    label = ROAD_TYPE_SHORT.get(hwy_type, hwy_type)
    ax.scatter(
        df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
        c=color, s=40, alpha=0.85, edgecolors='black', linewidths=0.3,
        zorder=3
    )
    legend_handles.append(mpatches.Patch(color=color, label=f"{label} ({mask.sum()})"))

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('METR-LA Sensors by Road Classification (OpenStreetMap)', fontsize=14)
ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_spatial_map.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. Spatial Map Colored by Number of Lanes
# ============================================================
print("Plotting spatial map by lanes...")
fig, ax = plt.subplots(figsize=(12, 10))

lanes_known = df[df['lanes'] > 0]
lanes_unknown = df[df['lanes'] == 0]

scatter = ax.scatter(
    lanes_known['longitude'], lanes_known['latitude'],
    c=lanes_known['lanes'], cmap='YlOrRd', s=50, alpha=0.85,
    edgecolors='black', linewidths=0.3, vmin=1, vmax=6, zorder=3
)
if len(lanes_unknown) > 0:
    ax.scatter(
        lanes_unknown['longitude'], lanes_unknown['latitude'],
        c='gray', s=30, alpha=0.4, marker='x',
        label=f'Unknown ({len(lanes_unknown)})', zorder=2
    )

cbar = plt.colorbar(scatter, ax=ax, label='Number of Lanes')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('METR-LA Sensors by Number of Lanes (OpenStreetMap)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_lanes_map.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. Road Type Distribution Bar Chart
# ============================================================
print("Plotting road type distribution...")
fig, ax = plt.subplots(figsize=(12, 6))

type_counts = df['highway'].value_counts()
type_labels = [ROAD_TYPE_SHORT.get(t, t) for t in type_counts.index]
colors = [CATEGORY_COLORS.get(t, '#cccccc') for t in type_counts.index]

bars = ax.bar(type_labels, type_counts.values, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Road Type', fontsize=12)
ax.set_ylabel('Number of Sensors', fontsize=12)
ax.set_title('Distribution of Road Types Across METR-LA Sensors', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_type_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Lanes Distribution Histogram
# ============================================================
print("Plotting lanes distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

known_lanes = df[df['lanes'] > 0]['lanes']
unknown_count = (df['lanes'] == 0).sum()

lane_counts = known_lanes.value_counts().sort_index()
axes[0].bar(lane_counts.index.astype(int), lane_counts.values, color='#4363d8',
            edgecolor='black', linewidth=0.5)
for lane_val, count in zip(lane_counts.index, lane_counts.values):
    axes[0].text(int(lane_val), count + 0.5, str(count), ha='center', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Number of Lanes', fontsize=12)
axes[0].set_ylabel('Number of Sensors', fontsize=12)
axes[0].set_title('Lane Count Distribution (Known Values)', fontsize=13)
axes[0].grid(True, alpha=0.3, axis='y')

pie_data = [len(known_lanes), unknown_count]
pie_labels = [f'Known ({len(known_lanes)})', f'Unknown ({unknown_count})']
axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90,
            colors=['#4363d8', '#cccccc'], textprops={'fontsize': 12})
axes[1].set_title('Lane Data Availability', fontsize=13)

plt.suptitle('Lane Information for METR-LA Sensors', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_lanes_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Road Type vs Average Speed Box Plot
# ============================================================
print("Plotting road type vs traffic speed...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

type_order = df.groupby('highway')['latitude'].count().sort_values(ascending=False).index.tolist()
type_order = [t for t in type_order if t != 'unknown']

grouped_data = []
grouped_labels = []
for hwy_type in type_order:
    mask = df['highway'] == hwy_type
    sensor_indices = np.where(mask.values)[0]
    speeds = avg_speed[sensor_indices]
    if len(speeds) > 0:
        grouped_data.append(speeds)
        grouped_labels.append(ROAD_TYPE_SHORT.get(hwy_type, hwy_type))

bp = axes[0].boxplot(grouped_data, labels=grouped_labels, patch_artist=True)
for patch, hwy_type in zip(bp['boxes'], type_order):
    patch.set_facecolor(CATEGORY_COLORS.get(hwy_type, '#cccccc'))
axes[0].set_ylabel('Average Speed (mph)', fontsize=12)
axes[0].set_xlabel('Road Type', fontsize=12)
axes[0].set_title('Average Traffic Speed by Road Type', fontsize=13)
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(True, alpha=0.3, axis='y')

lanes_order = sorted(df[df['lanes'] > 0]['lanes'].unique())
lanes_data = []
lanes_labels = []
for lane_val in lanes_order:
    mask = df['lanes'] == lane_val
    sensor_indices = np.where(mask.values)[0]
    speeds = avg_speed[sensor_indices]
    if len(speeds) > 0:
        lanes_data.append(speeds)
        lanes_labels.append(f'{int(lane_val)} lanes')

bp2 = axes[1].boxplot(lanes_data, labels=lanes_labels, patch_artist=True)
cmap = plt.cm.YlOrRd
for i, patch in enumerate(bp2['boxes']):
    frac = (lanes_order[i] - min(lanes_order)) / max(1, (max(lanes_order) - min(lanes_order)))
    patch.set_facecolor(cmap(0.3 + 0.6 * frac))
axes[1].set_ylabel('Average Speed (mph)', fontsize=12)
axes[1].set_xlabel('Number of Lanes', fontsize=12)
axes[1].set_title('Average Traffic Speed by Lane Count', fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Road Features vs Traffic Speed', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_speed_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Adjacency Matrix Heatmap with Road Type Overlay
# ============================================================
print("Plotting network graph with road types...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

adj_viz = np.array(adj_mx)
adj_viz[adj_viz > 0] = 1
im = axes[0].imshow(adj_viz, cmap='Blues', aspect='equal', interpolation='nearest')
axes[0].set_title('Sensor Adjacency Matrix', fontsize=13)
axes[0].set_xlabel('Sensor Index', fontsize=11)
axes[0].set_ylabel('Sensor Index', fontsize=11)
plt.colorbar(im, ax=axes[0], fraction=0.046, label='Connection')

highway_encoded = df['highway_encoded'].values
n = len(highway_encoded)
road_sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if highway_encoded[i] == highway_encoded[j]:
            road_sim[i, j] = 1

im2 = axes[1].imshow(road_sim, cmap='Set3', aspect='equal', interpolation='nearest')
axes[1].set_title('Road Type Similarity Matrix\n(Same Color = Same Road Type)', fontsize=13)
axes[1].set_xlabel('Sensor Index', fontsize=11)
axes[1].set_ylabel('Sensor Index', fontsize=11)
plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Same Road Type')

plt.suptitle('Network Structure vs Road Classification', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'road_network_graph.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Quantitative Summary
# ============================================================
print("\n=== Road Feature Summary ===")
print(f"Total sensors: {len(df)}")
print(f"\nRoad type distribution:")
for hwy_type in type_order:
    count = (df['highway'] == hwy_type).sum()
    mask = df['highway'] == hwy_type
    indices = np.where(mask.values)[0]
    if len(indices) > 0:
        mean_speed = avg_speed[indices].mean()
        pct = 100 * count / len(df)
        label = ROAD_TYPE_SHORT.get(hwy_type, hwy_type)
        print(f"  {label:20s}: {count:3d} sensors ({pct:5.1f}%) avg speed: {mean_speed:.1f} mph")

print(f"\nLanes distribution:")
for lane_val in sorted(df[df['lanes'] > 0]['lanes'].unique()):
    count = (df['lanes'] == lane_val).sum()
    mask = df['lanes'] == lane_val
    indices = np.where(mask.values)[0]
    if len(indices) > 0:
        mean_speed = avg_speed[indices].mean()
        pct = 100 * count / len(df)
        print(f"  {int(lane_val)} lanes: {count:3d} sensors ({pct:5.1f}%) avg speed: {mean_speed:.1f} mph")

unknown_count = (df['lanes'] == 0).sum()
print(f"  Unknown: {unknown_count:3d} sensors ({100*unknown_count/len(df):5.1f}%)")

print(f"\nAll visualizations saved to {FIGURE_DIR}/")
print("  - road_spatial_map.png")
print("  - road_lanes_map.png")
print("  - road_type_distribution.png")
print("  - road_lanes_distribution.png")
print("  - road_speed_correlation.png")
print("  - road_network_graph.png")