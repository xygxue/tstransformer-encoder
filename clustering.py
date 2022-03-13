import json
import pandas as pd
import numpy as np
import torch

from datasets.dataset import padding_mask
from models.ts_transformer import TSTransformerEncoder, TSTransformerEncoderOutput
from utils.utils import load_model


config_path = 'experiments/A0000008261_pretrained_2022-03-11_19-15-21_jwM/configuration.json'
model_path = 'experiments/A0000008261_pretrained_2022-03-11_19-15-21_jwM/checkpoints/model_best.pth'
best_pred_path = 'experiments/A0000008261_pretrained_2022-03-11_19-15-21_jwM/predictions/best_predictions.npz'
train_csv_path = 'data_exp/TRAIN.csv'

train = pd.read_csv(train_csv_path, index_col='account_id')


trans_acc_grp_unnorm = train.groupby(['account_id'])
features = []
col_num = train.shape[1] - 1
for k, v in trans_acc_grp_unnorm:
    array = torch.tensor(v.values)
    # append resized array to list
    features.append(array)

X = torch.zeros(4500, 675, 12)  # (batch_size, padded_length, feat_dim)
lengths = [a.shape[0] for a in features]


for i in range(4500):
    end = min(lengths[i], 675)
    X[i, :end, :] = features[i][:end, :]

X_batch = torch.split(X, 30)

pad = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=675)

with open(config_path, "r") as config_file:
    config = json.loads(config_file.read())

model = TSTransformerEncoderOutput(12, 675, config['d_model'], config['num_heads'],
                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                             norm=config['normalization_layer'], freeze=config['freeze'])
pre_model = load_model(model, model_path)
model_train = model.train()
output = []
for batch in X_batch:
    output_batch = model_train(batch.to('cpu'), pad)
    output = torch.cat((output, output_batch), 0)