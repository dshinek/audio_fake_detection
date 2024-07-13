import torch
from torch.utils.data import Dataset
import numpy as np
from src.config import Config
import random

class Train_CustomDataset(Dataset):
    def __init__(self, mfcc, label, real_data_list, fake_data_list, noise_data_list):
        self.mfcc = mfcc
        self.label = label
        self.real_data = real_data_list
        self.fake_data = fake_data_list
        self.noise_data = noise_data_list

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        feature = torch.from_numpy(np.load(self.mfcc[index]))
        if self.label is None:
            return feature
        random_number = random.randint(0,2) #random 0 means real, 1 means fake, 2 means noise
        
        #which data to add / fake data ? real data ? noise data ?

        if random_number == 0:
            add_feature = torch.from_numpy(np.load(random.choice(self.real_data)))
        elif random_number == 1:
            add_feature = torch.from_numpy(np.load(random.choice(self.fake_data)))
        elif random_number == 2:
            add_feature = torch.from_numpy(np.load(random.choice(self.noise_data)))

        syn_flg = random.randint(0,1)
        
        #How to make syn data / concat ? or merge ?

        if syn_flg == 0: #concat
            concat_type = random.randint(0,3)
            concat_point = random.choice([128*1,128*2,128*3,128*4,128*5,128*6,128*7])
            if concat_type == 0:
                feature = torch.concat((feature[:concat_point, :], add_feature[:1024-concat_point, :]),axis=0)
            elif concat_type == 1:
                feature = torch.concat((feature[concat_point:, :], add_feature[:concat_point, :]),axis=0)
            elif concat_type == 2:
                feature = torch.concat((feature[:concat_point, :], add_feature[concat_point:, :]),axis=0)
            elif concat_type == 3:
                feature = torch.concat((feature[concat_point:, :], add_feature[1024-concat_point:, :]),axis=0)  
        else: #merge
            feature = feature + add_feature

        fake = 0
        real = 0
        
        #labeling

        if self.label[index] == "real":
            if random_number == 0:
                real = 1
                fake = 0
            elif random_number == 1:
                real = 1
                fake = 1
            elif random_number == 2:
                real = 1
                fake = 0
        elif self.label[index] == "fake":
            if random_number == 0:
                real = 1
                fake = 1
            elif random_number == 1:
                real = 0
                fake = 1
            elif random_number == 2:
                real = 0
                fake = 1
        elif self.label[index] == "noise":
            if random_number == 0:
                real = 1
                fake = 0
            elif random_number == 1:
                real = 0
                fake = 1
            elif random_number == 2:
                real = 0
                fake = 0

        label_vector = torch.from_numpy(np.zeros(Config.N_CLASSES, dtype=float))
        label_vector[0] = fake
        label_vector[1] = real
        
        if self.label is not None:
            return feature, label_vector