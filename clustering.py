import json
import pandas as pd
import numpy as np
import torch

from datasets.dataset import padding_mask
from models.ts_transformer import TSTransformerEncoder, TSTransformerEncoderOutput
from utils.utils import load_model

sample_size = 4500
batch_size = 32
max_seq_len = 675
feat_dim = 12

config_path = 'experiments/3_pretrained_2022-03-17_14-46-30_eVA/configuration.json'
model_path = 'experiments/3_pretrained_2022-03-17_14-46-30_eVA/checkpoints/model_best.pth'
best_pred_path = 'experiments/3_pretrained_2022-03-17_14-46-30_eVA/predictions/best_predictions.npz'
train_csv_path = 'data_exp/TRAIN.csv'
latent_space_path = 'experiments/3_pretrained_2022-03-17_14-46-30_eVA/predictions/latent_space_output.pt'
train = pd.read_csv(train_csv_path, index_col='account_id')


trans_acc_grp_unnorm = train.groupby(['account_id'])
features = []
col_num = train.shape[1] - 1
for k, v in trans_acc_grp_unnorm:
    array = torch.tensor(v.values)
    # append resized array to list
    features.append(array)

X = torch.zeros(sample_size, max_seq_len, feat_dim)  # (sample_size, padded_length, feat_dim)
lengths = [a.shape[0] for a in features]


for i in range(sample_size):
    end = min(lengths[i], max_seq_len)
    X[i, :end, :] = features[i][:end, :]

X_batch = torch.split(X, batch_size) # (batch_size, padded_length, feat_dim)

pad = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_seq_len)
pad_batch = torch.split(pad, batch_size)

with open(config_path, "r") as config_file:
    config = json.loads(config_file.read())

model = TSTransformerEncoderOutput(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                             norm=config['normalization_layer'], freeze=config['freeze'])
pre_model = load_model(model.to('cpu'), model_path)
model_train = model.train()

output = torch.tensor([]).to('cpu')

for i in range(len(X_batch)):
    output_batch = model_train(X_batch[i].to('cpu'), pad_batch[i].to('cpu'))
    #output.append(output_batch)
    output = torch.cat((output, output_batch.detach().to('cpu')), dim=1)

torch.save(output, latent_space_path)
