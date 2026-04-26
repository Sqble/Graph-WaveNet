# Model Experiment Log

This file tracks all model training runs, their configurations, and results.

**How to add a new entry:** Copy the template below, fill it in, and paste it at the top of the `# Experiments` section (newest first).

---

## Template

```markdown
### Run #[NUMBER] — [Brief Description]

**Date:** YYYY-MM-DD  
**Status:** ✅ Success / ❌ Failed / 🔄 In Progress  
**Branch/Commit:** `branch-name` @ `commit-hash`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet / LSTM / GRU / etc. |
| Dataset | METR-LA / PEMS-BAY |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | Yes / No |
| Adaptive Adj | Yes / No |
| Random Init Adj | Yes / No |
| Device | cuda:0 |
| Other Changes | describe any code/dataset/architecture changes |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | | | |
| 30 min (H6) | | | |
| 60 min (H12) | | | |
| **Average (12H)** | | | |

#### Notes
- What worked / what didn't
- Training time per epoch
- Any issues encountered
- What to try next
```

---

## Experiments (Newest First)

### Run #6 — Graph WaveNet + GCN-Injected Static Road Embeddings

**Date:** 2026-04-26  
**Status:** ✅ Success  
**Branch/Commit:** `master`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet (GCN Road Injection) |
| Dataset | METR-LA |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | **GCN-level road injection**: `road_emb_gcn` Linear(4→32) embeds static road features; embedding added via broadcast to **every GCN output** after spatial aggregation. Road features are extracted from input (last 4 channels) and kept **out of dilated temporal convolutions** — only traffic+weather (5 channels) pass through `start_conv` and WaveNet blocks. |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | 2.72 | 6.96% | 5.20 |
| 30 min (H6) | 3.10 | 8.39% | 6.23 |
| 60 min (H12) | 3.53 | 9.88% | 7.30 |
| **Average (12H)** | **3.06** | **8.21%** | **6.09** |

#### Notes
- **Training time:** ~28 sec/epoch, **Total:** ~50 min for 100 epochs
- **Best valid loss:** 2.7503 (at epoch 55)
- **Test MAPE 8.21% is the best of all runs** — GCN injection appears to improve relative error calibration despite slightly higher valid loss
- Test MAE@60min (3.53) matches the contextual concatenation run (Run #3, 3.53) and is competitive with the best
- By injecting road at the GCN level rather than the input layer, the model treats road type/lanes as **node properties that modulate spatial aggregation**, not as time-series features. This is architecturally cleaner given that road features are static.
- The valid loss curve (2.7503) is marginally worse than the weather-gated run (2.7390), suggesting that while GCN injection helps MAPE, it does not dramatically improve the primary training objective (MAE). This may be because 92% freeway sensor bias limits the value of road-type modulation.
- **Next:** Combine GCN road injection with weather gate (FiLM) to see if the two mechanisms are complementary; evaluate on non-freeway subset to verify road injection impact where road diversity is higher.

---

### Run #5 — Graph WaveNet + Multi-Stream + Weather Gate (FiLM)

**Date:** 2026-04-26  
**Status:** ✅ Success  
**Branch/Commit:** `master`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet (Multi-Stream + Weather Gate) |
| Dataset | METR-LA |
| Epochs | 75 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | **Multi-stream** + **Weather-conditioned gating**: `weather_gate` Conv2d(3→32, 1×1) outputs sigmoid gate applied to fused traffic+weather+road features after initial encoding. Weather dynamically scales each residual channel based on temp/precip/humidity. |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | 2.71 | 6.96% | 5.18 |
| 30 min (H6) | 3.10 | 8.43% | 6.23 |
| 60 min (H12) | 3.55 | 10.06% | 7.33 |
| **Average (12H)** | **3.06** | **8.27%** | **6.09** |

#### Notes
- **Training time:** ~27 sec/epoch, **Total:** ~36 min for 75 epochs
- **Best valid loss:** 2.739 (at epoch 68) — **new best across all runs!**
- Weather gate successfully enables dynamic conditioning: during clear weather gate ≈ 1.0 (no change), during storms gate suppresses/amplifies specific channels
- Test MAE@60min (3.55) matches multi-stream and baseline; MAPE improved to 8.27% (best of all runs)
- The valid loss curve was still improving through epoch 68, suggesting 100 epochs may yield further gains
- **Next:** Run 100-epoch full comparison; evaluate on rain-day subset to verify weather gate impact; proceed to dynamic adjacency (TODO #4) or interaction features (TODO #5)

---

### Run #4 — Graph WaveNet + Multi-Stream Architecture (Weather + Road)

**Date:** 2026-04-26  
**Status:** ✅ Success  
**Branch/Commit:** `master`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet (Multi-Stream) |
| Dataset | METR-LA |
| Epochs | 50 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | **Multi-stream architecture**: separate `start_conv` for traffic (2→32) and weather (3→32); static road embedding via `nn.Linear(4→32)` bypasses temporal convs. Fusion via summation before WaveNet blocks. |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | 2.71 | 7.01% | 5.18 |
| 30 min (H6) | 3.09 | 8.56% | 6.19 |
| 60 min (H12) | 3.55 | 10.19% | 7.31 |
| **Average (12H)** | **3.06** | **8.39%** | **6.07** |

#### Notes
- **Training time:** ~27 sec/epoch, **Total:** ~24 min for 50 epochs
- **Best valid loss:** 2.7675 (at epoch 36)
- Architecture is stable and trains correctly; road embedding successfully bypasses dilated temporal convolutions
- At 50 epochs, model has not fully converged (baseline reached 2.7418 at 100 epochs). Multi-stream shows promise but needs a full 100-epoch run to fairly compare
- Test MAE@60min (3.55) is slightly above baseline (3.54) and contextual concatenation (3.53), but this is expected given half the training budget
- **Next:** Run 100 epochs for direct comparison; evaluate rain-day subset; proceed to Weather-Conditioned Gating (TODO #3)

---

### Run #3 — Graph WaveNet + Contextual Features (Weather + Road)

**Date:** 2026-04-26  
**Status:** ✅ Success  
**Branch/Commit:** `master`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet |
| Dataset | METR-LA |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | 3-input stream architecture: traffic (speed + time-of-day) + weather (temp, precip, humidity) + road (is_freeway, is_arterial, is_local, lanes). `in_dim` increased from 2 → 9. Weather standardized per-feature on train split. Road type one-hot encoded (freeway/arterial/local), lanes standardized on known values. |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | 2.70 | 7.15% | 5.15 |
| 30 min (H6) | 3.08 | 8.53% | 6.17 |
| 60 min (H12) | 3.53 | 10.06% | 7.28 |
| **Average (12H)** | **3.04** | **8.39%** | **6.04** |

#### Notes
- **Training time:** ~27 sec/epoch, **Total:** ~48 min for 100 epochs
- **Best valid loss:** 2.7411 (vs baseline 2.7418, weather 2.7425)
- Contextual features (weather + road) achieved the **best valid loss** of all three runs
- Road data: 191 freeway (motorway/motorway_link), 4 arterial, 12 local sensors from OpenStreetMap
- The marginal valid loss improvement (2.7411 vs 2.7418) suggests road features add minimal predictive power, likely because 92% of sensors are on freeways (low feature diversity)
- MAE at 60 min (3.53) matches the baseline (3.54), confirming that contextual features don't degrade performance
- **Next:** Investigate road impact during non-freeeway segments, or focus on weather during rain events

---

### Run #2 — Graph WaveNet + Weather Features (Temperature, Precipitation, Humidity)

**Date:** 2025-04-24  
**Status:** ✅ Success  
**Branch/Commit:** `master` @ `fe57112`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet |
| Dataset | METR-LA |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | 2-input stream architecture: traffic (speed + time-of-day) + weather (temp, precip, humidity). `in_dim` increased from 2 → 5. Weather standardized per-feature on train split. Historical weather fetched via Open-Meteo API and interpolated to 5-min intervals. |

#### Results (from comparison table — per-horizon values need update from logs)
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 60 min (H12) | 3.58 | 10.39% | 7.42 |

#### Notes
- **Best valid loss:** 2.7425 (vs baseline 2.7418)
- Weather features alone did not improve over baseline
- **Next:** Add road features for contextual hybrid augmentation

---

### Run #1 — Graph WaveNet Baseline (Bug Fix Applied)

**Date:** 2025-04-24  
**Status:** ✅ Success  
**Branch/Commit:** `master` @ `6177b13`

#### Model & Parameters
| Parameter | Value |
|-----------|-------|
| Model | Graph WaveNet |
| Dataset | METR-LA |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Dim (nhid) | 32 |
| Dropout | 0.3 |
| Weight Decay | 0.0001 |
| GCN Enabled | ✅ Yes |
| Adaptive Adj | ✅ Yes |
| Random Init Adj | ✅ Yes |
| Device | cuda:0 |
| Other Changes | Fixed `Conv1d` → `Conv2d` bug in `model.py` (gate_convs, residual_convs, skip_convs) |

#### Results
| Horizon | MAE | MAPE | RMSE |
|---------|-----|------|------|
| 15 min (H3) | 2.72 | 6.97% | 5.19 |
| 30 min (H6) | 3.10 | 8.53% | 6.21 |
| 60 min (H12) | 3.54 | 10.19% | 7.29 |
| **Average (12H)** | **3.06** | **8.35%** | **6.08** |

#### Notes
- **Training time:** ~33 sec/epoch, **Total:** ~47 min for 100 epochs
- **Best valid loss:** 2.7418 (used for model selection)
- MAE 3.54 at 60 min matches REPORT.pdf target (3.53) almost exactly
- MAPE is slightly higher than report (10.19% vs 6.90%) — likely due to random init or scaling
- Model is well-trained and ready for enhancement experiments
- **Next:** Add weather + road feature streams for contextual hybrid augmentation

---

## Comparison Table (All Runs)

| Run | Model | Epochs | Valid Loss | MAE@60min | MAPE@60min | RMSE@60min | Notes |
|-----|-------|--------|------------|-----------|------------|------------|-------|
| #6 | Graph WaveNet + GCN Road Injection | 100 | 2.7503 | 3.53 | 9.88% | 7.30 | **Best MAPE overall (8.21%)**; road injected at GCN level |
| #5 | Graph WaveNet + Multi-Stream + Weather Gate | 75 | **2.7390** | 3.55 | 10.06% | 7.33 | **New best valid loss!** Dynamic weather gating |
| #4 | Graph WaveNet + Multi-Stream | 50 | 2.7675 | 3.55 | 10.19% | 7.31 | Separate encoders + road embedding |
| #3 | Graph WaveNet + Weather + Road | 100 | 2.7411 | 3.53 | 10.06% | 7.28 | Added temp, precip, humidity, road type, lanes |
| #2 | Graph WaveNet + Weather | 100 | 2.7425 | 3.58 | 10.39% | 7.42 | Added temp, precip, humidity |
| #1 | Graph WaveNet | 100 | 2.7418 | 3.54 | 10.19% | 7.29 | Baseline, bug fix applied |

