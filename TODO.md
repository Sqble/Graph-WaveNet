# Graph WaveNet Traffic Forecasting — Project TODO

Based on the Comparative Analysis Report (REPORT.pdf).

---

## ✅ COMPLETED

### Bug Fix
- [x] Fix Conv1d → Conv2d bug in `model.py` (`gate_convs`, `residual_convs`, `skip_convs`)

### Baseline Training (100 epochs)
- [x] Train Graph WaveNet to convergence on METR-LA
  - Best valid loss: **2.7418**
  - 15 min: MAE 2.72, MAPE 6.97%, RMSE 5.19
  - 30 min: MAE 3.10, MAPE 8.53%, RMSE 6.21
  - 60 min: MAE 3.54, MAPE 10.19%, RMSE 7.29
  - **Avg (12H): MAE 3.06, MAPE 8.35%, RMSE 6.08**
  - Saved: `best_model/gwn_baseline_metr_la_best.pth`
  - Logged as **Run #1** in `EXPERIMENTS.md`

### Weather Data Collection (Stream 2)
- [x] Fetch historical weather data from Open-Meteo API for all 207 sensors (Mar–Jun 2012)
  - Features: `temperature_2m`, `precipitation`, `relative_humidity_2m`
  - Saved: `data/METR-LA/weather.npz` (34272, 207, 3)
  - Cache: `data/weather_cache/` (per-sensor JSON)
- [x] Interpolate hourly → 5-minute resolution (`fetch_weather.py`)
- [x] Preprocess and align with METR-LA timestamps
- [x] Standardize weather features per-feature on train split
- [x] Visualize weather data (`visualize_weather.py`)
  - `figures/weather_timeseries.png`
  - `figures/weather_spatial.png`
  - `figures/weather_distributions.png`
  - `figures/weather_traffic_comparison.png`
  - `figures/weather_correlation.png`

### Weather-Enhanced Model & Training (2-input)
- [x] Modify Graph WaveNet for weather input stream
  - `in_dim` increased: 2 → 5 (traffic speed + time-of-day + 3 weather features)
  - Created `train_weather.py` with sliding-window weather loader
  - Weather concatenated to traffic features at input layer
  - Core convolutions unchanged (modular design)
- [x] Train weather-enhanced model (100 epochs)
  - Best valid loss: **2.7425** (vs baseline 2.7418)
  - 15 min: MAE 2.72, MAPE 7.14%, RMSE 5.19
  - 30 min: MAE 3.10, MAPE 8.57%, RMSE 6.23
  - 60 min: MAE 3.58, MAPE 10.39%, RMSE 7.42
  - **Avg (12H): MAE 3.07, MAPE 8.50%, RMSE 6.13**
  - Saved: `best_model/gwn_weather_enhanced_best.pth`
  - Logged as **Run #2** in `EXPERIMENTS.md`

### Comparative Analysis
- [x] Document baseline and weather-enhanced results in `EXPERIMENTS.md`
- [x] Comparison table (Run #1 vs Run #2)
- [x] **Key finding:** Weather features did **not** improve average forecasting performance over baseline (near-zero linear correlation between weather and traffic speed)

---

## 🔄 IN PROGRESS

### Static Road Features (Stream 3)
- [ ] Collect road classification data for the 207 sensors
  - Road category (freeway, arterial, etc.)
  - Number of lanes at each sensor location
- [ ] Encode as node-level features (one-hot or embedding)

### Visualization & Reporting
- [ ] Performance comparison bar chart (all 7 models from report + our best run)
- [ ] MAE/MAPE vs horizon line plots (15/30/60 min)
- [ ] Architecture diagram for the proposed 3-input model
- [ ] Update REPORT.pdf with:
  - Enhanced model architecture description
  - Training logs and convergence curves
  - Final test results table
  - Discussion on improvement magnitude (or lack thereof for weather-only)
  - Limitations and future work

---

## ⏳ NOT STARTED

### Full 3-Stream Architecture (traffic + weather + road)
- [ ] Modify model for 3-input architecture
  - Stream 1: 12-step historical speed (existing)
  - Stream 2: Weather features (done, needs road added)
  - Stream 3: Static road features (pending data collection)
- [ ] Concatenate all 3 streams at input layer
- [ ] Train full contextual hybrid model (100 epochs)
- [ ] Evaluate and compare to baseline + weather-only

### Future Work
- [ ] Real-time incident data integration (accidents, lane closures)
- [ ] Transfer learning across different cities/datasets
- [ ] Empirical validation under adverse weather conditions
- [ ] Investigate weather impact specifically during rain days (subset analysis)

---

## Key Deliverables
1. ✅ Working baseline (`best_model/gwn_baseline_metr_la_best.pth`)
2. ✅ Weather-enhanced model (`best_model/gwn_weather_enhanced_best.pth`)
3. ⏳ Full contextual model with weather + road features (pending road data)
4. 🔄 Comparative results (baseline vs weather done; road pending)
5. 🔄 Updated report with results and analysis
6. ✅ Experiment log (`EXPERIMENTS.md`)
7. ✅ Weather data pipeline (`fetch_weather.py`, `visualize_weather.py`, `train_weather.py`)