from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import torchvision.io
import torchvision.transforms.functional
from skimage.transform import resize

def transformation_eurosat(target):
    target_tensor = torch.zeros(10)
    target_tensor[target] = 1
    return  target_tensor

class Train_Dataset_EuroSat(Dataset):
    def __init__(self, dataset : Dataset):
        self.train = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels\\Eurosat_idx_train.csv")
        self.dataset_ = dataset

    def __getitem__(self, index):
        x,y = self.dataset_[self.train.iloc[index]["idx"]]
        x = x*2 - 1
        return x ,y
    
    def __len__(self):
        return len(self.train)
    
class Test_Dataset_EuroSat(Dataset):
    def __init__(self, dataset : Dataset):
        self.test = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels\\Eurosat_idx_test.csv")
        self.dataset_ = dataset

    def __getitem__(self, index):
        x,y = self.dataset_[self.test.iloc[index]["idx"]]
        x = x*2 - 1
        return x ,y
    
    def __len__(self):
        return len(self.test)