import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from .utils import multiLabel_AUC
from .config import Config
import os
import json

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def train(model, optimizer, train_loader, val_loader, device, config):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    train_loss_list = []
    val_loss_list = []
    val_score_list = []
    
    for epoch in range(1, config.N_EPOCHS+1):
        with open("./expr/" + Config.EXPR_NAME + "_logging.json", "r") as json_file:
            expr_log = json.load(json_file)
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            
            output = model(features.to("cuda"))
            labels = labels.float().to("cuda")
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }

        ckpt_path = "./expr/" + Config.EXPR_NAME + "/ckpt"
        save_checkpoint(checkpoint, os.path.join(ckpt_path, f"./checkpoint_epoch_{epoch}.pth.tar"))

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        expr_log["Loss"]["Epoch_" + str(epoch)] = {}
        expr_log["Loss"]["Epoch_" + str(epoch)]["train_loss"] = str(_train_loss)
        expr_log["Loss"]["Epoch_" + str(epoch)]["val_loss"] = str(_val_loss)
        expr_log["Loss"]["Epoch_" + str(epoch)]["val_auc"] = str(_val_score)
        
        with open("./expr/" + Config.EXPR_NAME + "_logging.json", 'w') as outfile:
            json.dump(expr_log, outfile, indent=4)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

        train_loss_list.append(_train_loss)
        val_loss_list.append(_val_loss)
        val_score_list.append(_val_score)
    
    return best_model, train_loss_list, val_loss_list, val_score_list

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def checkpoint_val(ckpt_path,model,optimizer, val_loader, device):
    criterion = nn.BCELoss().to(device)
    train_loss_list = []
    val_loss_list = []
    val_score_list = []
    
    for epoch in range(1, Config.N_EPOCHS+1):
        
        if os.path.isfile(os.path.join(ckpt_path, f"checkpoint_epoch_{epoch}.pth.tar")):
            ckpt_model, optimizer, start_epoch, _train_loss = load_checkpoint(os.path.join(ckpt_path, f"checkpoint_epoch_{epoch}.pth.tar"), model, optimizer)
            ckpt_model.to("cuda")
            ckpt_model.eval()
            _val_loss, _val_score = validation(ckpt_model, criterion, val_loader, device)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')

            train_loss_list.append(_train_loss)
            val_loss_list.append(_val_loss)
            val_score_list.append(_val_score)
        else:
            continue
    
    return train_loss_list, val_loss_list, val_score_list