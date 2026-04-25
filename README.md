# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).



<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3.11 or lower
- see `requirements.txt`


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```

## Experiment Tracking

Every model run should be logged in [`EXPERIMENTS.md`](./EXPERIMENTS.md). This keeps a permanent record of:
- What was tried
- Hyperparameters used
- Final scores (MAE, MAPE, RMSE) for each horizon
- Whether it worked, failed, or is in progress

### How to Add a New Result

1. Open `EXPERIMENTS.md`
2. Copy the template at the top
3. Fill in your run details and scores
4. Add your row to the **Comparison Table** at the bottom
5. Commit the update with your code changes

> **Tip:** The template is designed to be copy-pasted. Keep the newest experiments at the top of the `# Experiments` section.

### Current Best Result

| Model | MAE@60min | MAPE@60min | Notes |
|-------|-----------|------------|-------|
| Graph WaveNet | 3.54 | 10.19% | Baseline (100 epochs, bug fix applied) |




