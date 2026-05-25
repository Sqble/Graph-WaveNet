# Short-Term Traffic Forecasting Final Project

This repository contains the Graph WaveNet work for short-term traffic forecasting on METR-LA, including contextual extensions for weather and road metadata.

## Project Focus

- Forecast traffic speed across METR-LA sensors using spatial-temporal graph neural networks.
- Compare baseline Graph WaveNet against contextual variants that add weather and road attributes.
- Keep static road attributes out of the temporal convolution path when possible, treating them as node-level spatial context.
- Track model runs, scores, and follow-up ideas in `EXPERIMENTS.md`.

## Current Best Run

| Run | Model | Epochs | Avg MAE | Avg MAPE | Avg RMSE | MAE@60min | MAPE@60min | RMSE@60min |
|-----|-------|--------|---------|----------|----------|-----------|------------|------------|
| #6 | Graph WaveNet + GCN road injection | 100 | 3.06 | 8.21% | 6.09 | 3.53 | 9.88% | 7.30 |

## Main Artifacts Kept Locally

The local workspace contains final reports, slide decks, generated figures, notebooks, synchronized HTML output, and video recordings. Those files are intentionally not all committed because several are large generated artifacts better suited for releases, cloud storage, or Git LFS.

Small figures and reproducible source files are the right candidates for normal Git commits. Large datasets, videos, exported HTML, and ZIP archives should stay out of the repository unless Git LFS is configured.

## Clean Publication Checklist

- Keep source code, scripts, experiment logs, and concise documentation in Git.
- Keep raw datasets and generated training data outside Git or under Git LFS.
- Add final reports/decks through GitHub Releases or cloud links if they are needed for sharing.
- Keep `EXPERIMENTS.md` as the canonical run history.
