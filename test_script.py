import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Train_CustomDataset
from src.model import MLP
from src.utils import seed_everything, get_data_path
from src.train import load_checkpoint
from src.inference import inference

seed_everything(Config.SEED)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
real_data_list = []
fake_data_list = []
noise_data_list = []

model = MLP()

optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LR)

infer_model, optimizer, epoch, loss = load_checkpoint("./expr/dynamic_dataset_including_kaggle_data_with_ast_model/ckpt/checkpoint_epoch_15.pth.tar",model,optimizer)

# Inference on test set
test = pd.read_csv('./data/test.csv')
test_wav = get_data_path(test, False)
test_dataset = Train_CustomDataset(test_wav, None,real_data_list,fake_data_list,noise_data_list)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

preds = inference(infer_model, test_loader, device)

# # Prepare submission
submit = pd.read_csv('./data/sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.to_csv(f'./expr/{Config.EXPR_NAME}_15epoch.csv', index=False)