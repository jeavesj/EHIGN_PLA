
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from graph_constructor import GraphDataset, collate_fn
from EHIGN import DTIPredictor
from utils import *

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')

import argparse
import time

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    time_list = []
    for data in dataloader:
        if data is None:
            continue
        t0 = time.time()
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2
            t1 = time.time()
            t_inf = (t1 - t0)/float(label.size(0))
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
        time_list.extend([t_inf]*label.size(0))

    if len(pred_list) == 0:
        raise RuntimeError('No valid graphs were loaded. Check that .dgl files exist.')
    
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return pred, rmse, pr, time_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/toy_set')
    parser.add_argument('--data_csv', type=str, default='./data/toy_examples.csv')
    parser.add_argument('--output_csv', type=str, required=False)
    parser.add_argument('--device', type=str, default='cpu', help='Options are cpu and gpu')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    data_csv = args.data_csv
    data_df = pd.read_csv(data_csv)

    data_set = GraphDataset(data_dir, data_df, graph_type='Graph_EHIGN', create=False)

    # mask of complexes that actually have graph files
    valid_mask = [os.path.exists(p) for p in data_set.graph_paths]

    data_loader = DataLoader(data_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    
    if args.device == 'gpu':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    load_model_dict(model, './model/20230120_135757_EHIGN_repeat0/model/epoch-144, train_loss-0.5772, train_rmse-0.7598, valid_rmse-1.1799, valid_pr-0.7718.pt')

    pred, rmse, pr, time_list = val(model, data_loader, device)

    msg = "rmse-%.4f, pr-%.4f," \
                % (rmse, pr)
    print(msg)
    
    if args.output_csv:
        n_valid = sum(valid_mask)
        if len(pred) != n_valid or len(time_list) != n_valid:
            raise RuntimeError(
                f'Mismatch between valid graphs ({n_valid}) and predictions ({len(pred)}) or timings ({len(time_list)})'
            )

        # drop complexes without graphs so labels and preds stay aligned
        data_df_valid = data_df[valid_mask].copy()
        data_df_valid['prediction'] = pred
        data_df_valid['time_inf_s'] = time_list
        data_df_valid.to_csv(args.output_csv, index=False)


# %%
