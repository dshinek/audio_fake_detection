import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Train_CustomDataset
from src.model import MLP
from src.utils import seed_everything, get_data
from src.train import train
from src.inference import inference
from sklearn.model_selection import train_test_split
import json

seed_everything(Config.SEED)
os.makedirs("./expr/" + Config.EXPR_NAME,exist_ok=True)
os.makedirs("./expr/" + Config.EXPR_NAME + "/ckpt",exist_ok=True)
data = {"expr_name": Config.EXPR_NAME, "expr_description": Config.EXPR_DESCRIPTION, "backbone_network": Config.BACKBONE_NAME,"SR": Config.SR, "BATCH_SIZE": Config.BATCH_SIZE, "Total_Epoch": Config.N_EPOCHS, "LR": Config.LR, "TRAIN_VAL_RATE": Config.TRAIN_VAL_RATE, "Loss": {}}

with open("./expr/" + Config.EXPR_NAME + "_logging.json", 'w') as outfile:
    json.dump(data, outfile,indent=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # Load data
df = pd.read_csv('./data/new_train_data_with_kaggle.csv')

real_data_list = []
fake_data_list = []
noise_data_list = []

for index, row in df.iterrows():
    if row["label"] == "real":
        real_data_list.append("./spectrogram/" + row["id"] + ".npy")
    elif row["label"] == "fake":
        fake_data_list.append("./spectrogram/" + row["id"] + ".npy")
    elif row["label"] == "noise":
        noise_data_list.append("./spectrogram/" + row["id"] + ".npy")

train_data, val_data, _, _ = train_test_split(df, df['label'], test_size=Config.TRAIN_VAL_RATE, random_state=Config.SEED)

# # Preprocess data
train_wav, train_labels = get_data(train_data, True)
val_wav, val_labels = get_data(val_data, True)

train_dataset = Train_CustomDataset(train_wav, train_labels,real_data_list,fake_data_list,noise_data_list)
val_dataset = Train_CustomDataset(val_wav, val_labels,real_data_list,fake_data_list,noise_data_list)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Initialize model
model = MLP()

optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LR)

# Train model
infer_model, train_loss_seq, val_loss_seq, val_score_seq = train(model, optimizer, train_loader, val_loader, device, Config)

# Inference on test set
test = pd.read_csv('./data/test.csv')
test_wav = get_data(test, False)
test_dataset = Train_CustomDataset(test_wav, None,real_data_list,fake_data_list,noise_data_list)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

preds = inference(infer_model, test_loader, device)

# # Prepare submission
submit = pd.read_csv('./data/sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.to_csv(f'./expr/{Config.EXPR_NAME}.csv', index=False)