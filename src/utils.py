import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .config import Config
from transformers import Wav2Vec2Model, ASTModel

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def pad_or_truncate_sequence(sequence, max_length):
    current_length = sequence.size(0)
    
    if current_length > max_length:
        excess_length = current_length - max_length
        start_index = excess_length // 2
        end_index = start_index + max_length
        sequence = sequence[start_index:end_index]
    else:
        padding_length = max_length - current_length
        sequence = F.pad(sequence, (0, padding_length), "constant", 0)
    
    return np.array(sequence)

def get_data(df, train_mode=True):
    features = []
    labels = []

    for _, row in tqdm(df.iterrows()):
        features.append("./spectrogram/"+str(row["id"] + ".npy"))
        if train_mode:
            label = row["label"]
            labels.append(label)

    if train_mode:
        return features, labels
    return features

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score