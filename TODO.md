# Graph WaveNet Traffic Forecasting — Project TODO

Based on the Comparative Analysis Report (REPORT.pdf), this is the complete task list for the project.

---

## PHASE 1: Baseline Training (IN PROGRESS)
- [ ] **Train Graph WaveNet baseline to convergence (100 epochs)** on METR-LA
  - Target from report: MAE ~3.53, MAPE ~6.90% at 60-min horizon
  - Current: Epoch 71, training well (~6.84% MAPE on iterations)
  - Wait for full 100 epochs, then capture test results
- [x] ~~Fix Conv1d -> Conv2d bug in model.py~~ ✅ Done

---

## PHASE 2: Data Collection — Contextual Features (REPORT Section VI)
The report proposes a **Contextual Hybrid Augmentation** approach. We need three input streams:

### Stream 2: Weather Data
- [ ] Collect weather features for LA area (March–June 2012) via meteorological APIs
  - Required fields: **precipitation**, **visibility**, **temperature**
  - Same 5-minute temporal resolution as METR-LA
  - Public APIs: NOAA, OpenWeatherMap historical, or similar
- [ ] Preprocess and align weather data with METR-LA timestamps
- [ ] Handle missing values (interpolation)

### Stream 3: Static Road Features
- [ ] Collect road classification data for the 207 sensors
  - **Road category** (freeway, arterial, etc.)
  - **Number of lanes** at each sensor location
- [ ] Encode as node-level features (one-hot or embedding)

---

## PHASE 3: Enhanced Model Implementation (REPORT Section VI.B)
- [ ] **Modify Graph WaveNet for multi-input architecture**
  - Input Stream 1: 12-step historical speed (existing)
  - Input Stream 2: Weather features (new)
  - Input Stream 3: Static road features (new)
- [ ] **Feature Fusion Layer**
  - Concatenate the 3 streams at the input layer
  - Pass through adaptive graph convolutional layers
  - Keep self-adaptive adjacency matrix intact
- [ ] Ensure model remains modular — weather/road features added post-hoc without changing core convolutions

---

## PHASE 4: Training & Evaluation
- [ ] Train enhanced model on METR-LA with contextual inputs
- [ ] Evaluate across forecasting horizons: **15 min, 30 min, 60 min**
- [ ] Metrics: **MAE, RMSE, MAPE** (same as baseline)
- [ ] Run for same 100 epochs for fair comparison

---

## PHASE 5: Comparative Analysis (REPORT Section V)
- [ ] **Baseline results** (temporal-only models from report):
  - LSTM: MAE 4.37 (60 min)
  - GRU: MAE 4.11 (60 min)
  - TCN: MAE 3.96 (60 min)
- [ ] **Spatiotemporal results** (from report):
  - DCRNN: MAE 3.60 (60 min)
  - STGCN: MAE 4.59 (60 min)
  - Graph WaveNet: MAE 3.53, MAPE 6.90% (60 min) ← **Our baseline target**
  - GMAN: MAE 2.80 (60 min)
- [ ] **Our enhanced model results**
- [ ] Document improvement over baseline Graph WaveNet

---

## PHASE 6: Visualization & Reporting
- [ ] Generate **performance comparison bar chart** (all 7 models + enhanced)
- [ ] Generate **MAE/MAPE vs horizon line plots** (15/30/60 min)
- [ ] Create **architecture diagram** for the proposed multi-input model
- [ ] Update **REPORT.pdf** with:
  - Enhanced model architecture description
  - Training logs and convergence curves
  - Final test results table
  - Discussion on improvement magnitude
  - Limitations and future work

---

## PHASE 7: Future Work (REPORT Section VII)
- [ ] Real-time incident data integration (accidents, lane closures)
- [ ] Transfer learning across different cities/datasets
- [ ] Empirical validation under adverse weather conditions

---

## Key Deliverables
1. Working Graph WaveNet baseline ✅ (bug fixed, training)
2. Enhanced Graph WaveNet with weather + road features
3. Comparative results table (8 models total)
4. Updated report with results and analysis
5. Clean codebase pushed to personal fork
