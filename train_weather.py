"""
Train Graph WaveNet with weather features as additional input stream.

Usage:
    # Quick test (2 epochs)
    python train_weather.py --device cuda:0 --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 2 --print_every 10 --save ./garage/metr_weather

    # Full 100-epoch run
    python train_weather.py --device cuda:0 --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 100 --print_every 10 --save ./garage/metr_weather
"""

import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--weather', type=str, default='data/METR-LA/weather.npz', help='weather data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=5, help='inputs dimension (2 traffic + 3 weather)')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./garage/metr_weather', help='save path')
parser.add_argument('--expid', type=int, default=2, help='experiment id')

args = parser.parse_args()


def load_weather_data(weather_path):
    """Load and window weather data to match traffic samples."""
    print("Loading weather data...")
    weather_raw = np.load(weather_path)['data']  # (34272, 207, 3)
    num_total, num_nodes, num_wfeat = weather_raw.shape
    
    # Create sliding windows: sample i covers timesteps [i, i+1, ..., i+11]
    # This matches the traffic data generation (x_offsets = [-11, ..., 0])
    num_samples = num_total - 12  # 34260? But traffic has 34249
    # Actually the traffic generation starts at t=11 and goes to t=num_total-12
    # So num_samples = (num_total - 12) - 11 = num_total - 23? No...
    # From generate_training_data.py: for t in range(min_t, max_t)
    # min_t = 11, max_t = num_total - 12 = 34260
    # So num_samples = 34260 - 11 = 34249
    
    # Sample i corresponds to t = 11 + i
    # x covers global timesteps [i, i+1, ..., i+11]
    
    num_samples = 34249  # known from traffic data
    weather_x = np.zeros((num_samples, 12, num_nodes, num_wfeat), dtype=np.float32)
    
    for i in range(num_samples):
        weather_x[i] = weather_raw[i:i+12, :, :]
    
    print(f"Weather windows: {weather_x.shape}")
    return weather_x


def combine_traffic_weather(data, weather_x):
    """
    Concatenate weather features with traffic input features.
    
    data: dict with x_train, x_val, x_test, y_train, y_val, y_test
    weather_x: (34249, 12, 207, 3)
    """
    # Match the traffic data split
    num_train = data['x_train'].shape[0]
    num_val = data['x_val'].shape[0]
    num_test = data['x_test'].shape[0]
    
    weather_train = weather_x[:num_train]
    weather_val = weather_x[num_train:num_train+num_val]
    weather_test = weather_x[num_train+num_val:]
    
    # Standardize weather (fit on train only)
    w_mean = weather_train.mean(axis=(0, 1, 2))  # (3,)
    w_std = weather_train.std(axis=(0, 1, 2))    # (3,)
    w_std[w_std == 0] = 1.0  # avoid division by zero
    
    def norm(w):
        return (w - w_mean) / w_std
    
    weather_train = norm(weather_train)
    weather_val = norm(weather_val)
    weather_test = norm(weather_test)
    
    print(f"Weather mean: {w_mean}, std: {w_std}")
    
    # Concatenate along feature dimension (last axis)
    data['x_train'] = np.concatenate([data['x_train'], weather_train], axis=-1)
    data['x_val'] = np.concatenate([data['x_val'], weather_val], axis=-1)
    data['x_test'] = np.concatenate([data['x_test'], weather_test], axis=-1)
    
    print(f"Combined x_train shape: {data['x_train'].shape}")
    print(f"Combined x_val shape: {data['x_val'].shape}")
    print(f"Combined x_test shape: {data['x_test'].shape}")
    
    return data


def load_dataset_with_weather(dataset_dir, weather_path, batch_size, valid_batch_size=None, test_batch_size=None):
    """Load traffic data and merge with weather windows."""
    # First load traffic data the standard way
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    scaler = util.StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Standardize only traffic speed (feature 0)
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    # Load and merge weather
    weather_x = load_weather_data(weather_path)
    data = combine_traffic_weather(data, weather_x)
    
    data['train_loader'] = util.DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = util.DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = util.DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def main():
    import os
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = load_dataset_with_weather(args.data, args.weather, args.batch_size, args.batch_size, args.batch_size)
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
                     adjinit)

    print("start training with weather features...", flush=True)
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
