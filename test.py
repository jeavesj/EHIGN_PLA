
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

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, pr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/toy_set')
    parser.add_argument('--data_csv', type=str, default='./data/toy_examples.csv')

    args = parser.parse_args()
    data_dir = args.data_dir
    data_csv = args.data_csv
    data_df = pd.read_csv(data_csv)

    data_set = GraphDataset(data_dir, data_df, graph_type='Graph_EHIGN', create=False)

    data_loader = DataLoader(data_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    load_model_dict(model, './model/20230120_135757_EHIGN_repeat0/model/epoch-144, train_loss-0.5772, train_rmse-0.7598, valid_rmse-1.1799, valid_pr-0.7718.pt')

    rmse, pr = val(model, data_loader, device)

    msg = "rmse-%.4f, pr-%.4f," \
                % (rmse, pr)
    print(msg)


# %%
