from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

class Bigearth(Dataset):
    def __init__(self):
        # self.X = torch.from_numpy(np.array(x_train)).type(torch.float)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc = enc.fit(y_train)
        self.y = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels.csv")
        self.database_len = 590_326#500
        self.start_generated = 590_327
        self.n_samples = self.database_len
    
    def __getitem__(self, index):
        self.X = plt.imread(f"C:\D\VS_programs_python\inzynierka\BigearthNet_png\{index}.png")
        return self.X, self.y.iloc[index-1].values[1:]
    
    def __len__(self):
        return self.n_samples
    
    def get_start_generated(self):
        return self.start_generated

class Bigearth_Pruned(Dataset):
    def __init__(self):
        # self.X = torch.from_numpy(np.array(x_train)).type(torch.float)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc = enc.fit(y_train)
        self.y = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels_cut.csv")
        self.database_len = len(self.y)#500#
        self.database_len_extended = 700
        self.start_generated = 590_327
        self.n_samples = self.database_len
    
    def __getitem__(self, index):
        self.X = plt.imread(f"C:\D\VS_programs_python\inzynierka\BigearthNet_png\{self.y.iloc[index-1].values[0]}.png")
        return self.X, self.y.iloc[index-1].values[1:]
    
    def get_y(self, index):
        return self.y.iloc[index-1].values[1:]

    def __len__(self):
        return self.n_samples
    
    def get_start_generated(self):
        return self.start_generated


class Dataloader(Dataset):
    def __init__(self, dataset : Dataset, test_size=0.2, random_seed = 23421):
        indexes = np.array([i for i in range(dataset.__len__())])
        self.dataset_ = dataset
        self.train, self.test = sklearn.model_selection.train_test_split(indexes, test_size=test_size, random_state=random_seed)

    def __getitem__(self, index):
        #TODO: add batches
        x,y = self.dataset_[self.train[index] + 1]
        return [x],[y]
    
    def __len__(self):
        return len(self.train)
    
    def get_test(self, index):
        return self.dataset_[self.test[index] + 1]
    
    def test_len(self):
        return len(self.test)