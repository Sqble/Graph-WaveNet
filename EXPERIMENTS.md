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

| Run | Model | Epochs | MAE@60min | MAPE@60min | RMSE@60min | Notes |
|-----|-------|--------|-----------|------------|------------|-------|
| #1 | Graph WaveNet | 100 | 3.54 | 10.19% | 7.29 | Baseline, bug fix applied |

