"""
Train Graph WaveNet with contextual features (weather + road).

Input streams:
  1. Traffic:  historical speed + time-of-day (2 features, from METR-LA npz)
  2. Weather:  temperature, precipitation, humidity (3 features, from weather.npz)
  3. Road:     is_freeway, is_arterial, is_local, lanes (4 features, from road_features.npz)

Total in_dim = 2 + 3 + 4 = 9

Usage:
    # Quick test (2 epochs)
    python train_contextual.py --device cuda:0 --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 2 --print_every 10 --save ./garage/metr_contextual

    # Full 100-epoch run
    python train_contextual.py --device cuda:0 --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 100 --print_every 10 --save ./garage/metr_contextual
"""

import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/METR-LA')
parser.add_argument('--weather', type=str, default='data/METR-LA/weather.npz')
parser.add_argument('--road', type=str, default='data/METR-LA/road_features.npz')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl')
parser.add_argument('--adjtype', type=str, default='doubletransition')
parser.add_argument('--gcn_bool', action='store_true')
parser.add_argument('--aptonly', action='store_true')
parser.add_argument('--addaptadj', action='store_true')
parser.add_argument('--randomadj', action='store_true')
parser.add_argument('--seq_length', type=int, default=12)
parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--in_dim', type=int, default=9, help='2 traffic + 3 weather + 4 road')
parser.add_argument('--num_nodes', type=int, default=207)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save', type=str, default='./garage/metr_contextual')
parser.add_argument('--expid', type=int, default=3)
parser.add_argument('--use_multi_stream', action='store_true', help='use separate encoders for traffic/weather/road')
parser.add_argument('--use_weather_gate', action='store_true', help='use weather-conditioned gating (FiLM-style)')

args = parser.parse_args()

ROAD_CATEGORIES = {
    'motorway': 'freeway',
    'motorway_link': 'freeway',
    'trunk': 'arterial',
    'trunk_link': 'arterial',
    'primary': 'arterial',
    'primary_link': 'arterial',
    'secondary': 'arterial',
    'secondary_link': 'arterial',
    'tertiary': 'arterial',
    'tertiary_link': 'arterial',
    'residential': 'local',
    'living_street': 'local',
    'service': 'local',
    'unclassified': 'local',
    'road': 'local',
    'path': 'local',
    'footway': 'local',
    'steps': 'local',
}

NUM_ROAD_FEATURES = 4  # is_freeway, is_arterial, is_local, lanes


def load_weather_data(weather_path):
    print("Loading weather data...")
    weather_raw = np.load(weather_path)['data']  # (34272, 207, 3)
    num_samples = 34249
    num_nodes = 207
    num_wfeat = weather_raw.shape[2]
    weather_x = np.zeros((num_samples, 12, num_nodes, num_wfeat), dtype=np.float32)
    for i in range(num_samples):
        weather_x[i] = weather_raw[i:i+12, :, :]
    print(f"Weather windows: {weather_x.shape}")
    return weather_x


def load_road_data(road_path, num_nodes):
    print("Loading road data...")
    road_data = np.load(road_path)
    highway_raw = road_data['highway_raw']  # (207,) strings
    lanes = road_data['lanes'].astype(np.float32)  # (207,)

    road_features = np.zeros((num_nodes, NUM_ROAD_FEATURES), dtype=np.float32)
    for i, hwy in enumerate(highway_raw):
        cat = ROAD_CATEGORIES.get(str(hwy), 'local')
        if cat == 'freeway':
            road_features[i, 0] = 1.0
        elif cat == 'arterial':
            road_features[i, 1] = 1.0
        else:
            road_features[i, 2] = 1.0
        road_features[i, 3] = lanes[i]

    # Standardize lanes (mean/std on known values only, 0 stays 0 for unknown)
    known_mask = lanes > 0
    lanes_mean = lanes[known_mask].mean()
    lanes_std = lanes[known_mask].std()
    if lanes_std == 0:
        lanes_std = 1.0
    road_features[known_mask, 3] = (lanes[known_mask] - lanes_mean) / lanes_std
    road_features[~known_mask, 3] = 0.0

    print(f"Road features shape: {road_features.shape}")
    freeway_count = int(road_features[:, 0].sum())
    arterial_count = int(road_features[:, 1].sum())
    local_count = int(road_features[:, 2].sum())
    print(f"  Freeway: {freeway_count}, Arterial: {arterial_count}, Local: {local_count}")
    print(f"  Lanes: mean={lanes_mean:.2f}, std={lanes_std:.2f}")
    return road_features


def load_dataset_with_contextual(dataset_dir, weather_path, road_path, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = util.StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    # Load and merge weather
    weather_x = load_weather_data(weather_path)
    num_train = data['x_train'].shape[0]
    num_val = data['x_val'].shape[0]
    num_test = data['x_test'].shape[0]

    weather_train = weather_x[:num_train]
    weather_val = weather_x[num_train:num_train+num_val]
    weather_test = weather_x[num_train+num_val:]

    w_mean = weather_train.mean(axis=(0, 1, 2))
    w_std = weather_train.std(axis=(0, 1, 2))
    w_std[w_std == 0] = 1.0
    weather_train = (weather_train - w_mean) / w_std
    weather_val = (weather_val - w_mean) / w_std
    weather_test = (weather_test - w_mean) / w_std
    print(f"Weather mean: {w_mean}, std: {w_std}")

    # Concatenate weather
    data['x_train'] = np.concatenate([data['x_train'], weather_train], axis=-1)
    data['x_val'] = np.concatenate([data['x_val'], weather_val], axis=-1)
    data['x_test'] = np.concatenate([data['x_test'], weather_test], axis=-1)

    # Load and tile road features across all timesteps
    num_nodes = data['x_train'].shape[2]
    road_features = load_road_data(road_path, num_nodes)  # (207, 4)

    # Tile: (num_samples, 12, 207, 4)
    road_train = np.tile(road_features[np.newaxis, np.newaxis, :, :], (num_train, 12, 1, 1))
    road_val = np.tile(road_features[np.newaxis, np.newaxis, :, :], (num_val, 12, 1, 1))
    road_test = np.tile(road_features[np.newaxis, np.newaxis, :, :], (num_test, 12, 1, 1))

    # Concatenate road features
    data['x_train'] = np.concatenate([data['x_train'], road_train], axis=-1)
    data['x_val'] = np.concatenate([data['x_val'], road_val], axis=-1)
    data['x_test'] = np.concatenate([data['x_test'], road_test], axis=-1)

    print(f"Combined x_train shape: {data['x_train'].shape}")
    print(f"Combined x_val shape: {data['x_val'].shape}")
    print(f"Combined x_test shape: {data['x_test'].shape}")
    print(f"Input: 2 traffic + 3 weather + 4 road = 9 features")

    data['train_loader'] = util.DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = util.DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = util.DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def main():
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = load_dataset_with_contextual(args.data, args.weather, args.road, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, use_multi_stream=args.use_multi_stream, traffic_dim=2, weather_dim=3, road_dim=4,
                     use_weather_gate=args.use_weather_gate)

    if args.use_multi_stream and args.use_weather_gate:
        print("start training with MULTI-STREAM + WEATHER GATE contextual features...", flush=True)
    elif args.use_multi_stream:
        print("start training with MULTI-STREAM contextual features (weather + road)...", flush=True)
    else:
        print("start training with contextual features (weather + road)...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))