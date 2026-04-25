"""
Visualize METR-LA weather data and traffic patterns.

Usage:
    python visualize_weather.py

Outputs:
    - figures/weather_timeseries.png
    - figures/weather_spatial.png
    - figures/weather_distributions.png
    - figures/weather_traffic_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Create output directory
os.makedirs('figures', exist_ok=True)

# Load data
print("Loading data...")
weather = np.load('data/METR-LA/weather.npz')['data']

# Traffic data for comparison
import util
dataloader = util.load_dataset('data/METR-LA', 64, 64, 64)
traffic = dataloader['x_train']  # (samples, 12, 207, 2)
scaler = dataloader['scaler']

# Generate timestamps
timestamps = [datetime(2012, 3, 1) + timedelta(minutes=5*i) for i in range(weather.shape[0])]

feature_names = ['Temperature (°C)', 'Precipitation (mm)', 'Relative Humidity (%)']
feature_keys = ['temperature', 'precipitation', 'humidity']

# ============================================
# 1. Time Series for Sample Sensors
# ============================================
print("Plotting time series...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Pick 5 diverse sensors (spread across LA)
sample_sensors = [0, 50, 100, 150, 200]
colors = plt.cm.tab10(np.linspace(0, 1, len(sample_sensors)))

for feat_idx, (ax, name, key) in enumerate(zip(axes, feature_names, feature_keys)):
    for i, sensor in enumerate(sample_sensors):
        ax.plot(timestamps, weather[:, sensor, feat_idx], 
                alpha=0.7, color=colors[i], label=f'Sensor {sensor}', linewidth=0.8)
    ax.set_ylabel(name, fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Highlight a rainy day if precipitation
    if feat_idx == 1:
        ax.set_ylim(0, max(2, weather[:, :, feat_idx].max()))

axes[-1].set_xlabel('Date', fontsize=11)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.suptitle('Weather Time Series — Sample Sensors (METR-LA, Mar–Jun 2012)', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig('figures/weather_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 2. Spatial Heatmaps at Specific Times
# ============================================
print("Plotting spatial heatmaps...")

# Load sensor coordinates
import pandas as pd
coords = pd.read_csv('data/sensor_graph/graph_sensor_locations.csv')

# Match sensor order
import pickle
sensor_ids, _, _ = pickle.load(open('data/sensor_graph/adj_mx.pkl', 'rb'), encoding='latin1')
id_to_latlon = {str(int(row['sensor_id'])): (row['latitude'], row['longitude']) 
                for _, row in coords.iterrows()}
lats = np.array([id_to_latlon[str(sid)][0] for sid in sensor_ids])
lons = np.array([id_to_latlon[str(sid)][1] for sid in sensor_ids])

# Pick 4 timepoints: morning, afternoon, rainy day, dry day
timepoints = [
    (1000, 'Day 3 ~10:00 AM'),
    (3000, 'Day 11 ~2:00 PM'),
    (15000, 'Rainy Day (Apr 15)'),
    (25000, 'Dry Day (May 20)')
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, (t_idx, title) in enumerate(timepoints):
    for row, feat_idx in enumerate([0, 2]):  # Temperature and Humidity
        ax = axes[row, col]
        values = weather[t_idx, :, feat_idx]
        scatter = ax.scatter(lons, lats, c=values, cmap='RdYlBu_r', s=20, alpha=0.8)
        ax.set_title(f'{title}\n{feature_names[feat_idx]}', fontsize=9)
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Spatial Weather Distribution Across LA Sensor Network', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/weather_spatial.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 3. Distribution Histograms
# ============================================
print("Plotting distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, feat_idx, name in zip(axes, range(3), feature_names):
    data = weather[:, :, feat_idx].flatten()
    ax.hist(data, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{name}\nμ={data.mean():.2f}, σ={data.std():.2f}', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Weather Feature Distributions (All Sensors, All Timesteps)', fontsize=13)
plt.tight_layout()
plt.savefig('figures/weather_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 4. Weather vs Traffic Speed Comparison
# ============================================
print("Plotting weather vs traffic...")

# Load traffic directly from HDF5 to match weather length (34272)
traffic_df = pd.read_hdf('data/metr-la.h5', key='/df')
traffic_full = traffic_df.values  # (34272, 207)

# Average across sensors for overview
avg_speed = traffic_full.mean(axis=1)  # (34272,)
avg_temp = weather[:, :, 0].mean(axis=1)  # (34272,)
avg_precip = weather[:, :, 1].mean(axis=1)

t = range(len(avg_speed))
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(t, avg_speed, color='green', linewidth=0.5, alpha=0.8)
axes[0].set_ylabel('Avg Speed (mph)', fontsize=11)
axes[0].set_title('Average Traffic Speed', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Temperature
axes[1].plot(t, avg_temp, color='red', linewidth=0.5, alpha=0.8)
axes[1].set_ylabel('Avg Temp (°C)', fontsize=11)
axes[1].set_title('Average Temperature', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Precipitation
axes[2].plot(t, avg_precip, color='blue', linewidth=0.5, alpha=0.8)
axes[2].set_ylabel('Avg Precip (mm)', fontsize=11)
axes[2].set_title('Average Precipitation', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('Time (5-min intervals)', fontsize=11)

plt.suptitle('Traffic Speed vs Weather Patterns Over Time', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig('figures/weather_traffic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 5. Correlation Matrix
# ============================================
print("Plotting correlations...")

# Use weather length as reference (34272 timesteps)
# Get traffic from HDF5 directly to match weather length
import pandas as pd
traffic_df = pd.read_hdf('data/metr-la.h5', key='/df')
traffic_full = traffic_df.values  # (34272, 207)

# Sample a subset for memory efficiency
sample_t = np.random.choice(weather.shape[0], 5000, replace=False)
sample_s = np.random.choice(weather.shape[1], 50, replace=False)

w_sample = weather[sample_t][:, sample_s, :]
t_sample = traffic_full[sample_t][:, sample_s]

# Flatten
w_flat = w_sample.reshape(-1, 3)
t_flat = t_sample.reshape(-1)

# Combine
data_matrix = np.column_stack([w_flat, t_flat])
labels = ['Temperature', 'Precipitation', 'Humidity', 'Traffic Speed']

corr = np.corrcoef(data_matrix.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

# Add text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', 
                      color='white' if abs(corr[i, j]) > 0.5 else 'black', fontsize=12)

plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Weather-Traffic Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('figures/weather_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved to figures/:")
print("  - weather_timeseries.png")
print("  - weather_spatial.png")
print("  - weather_distributions.png")
print("  - weather_traffic_comparison.png")
print("  - weather_correlation.png")
