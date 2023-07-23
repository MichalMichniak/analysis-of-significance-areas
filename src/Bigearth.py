from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Bigearth(Dataset):
    def __init__(self):
        # self.X = torch.from_numpy(np.array(x_train)).type(torch.float)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc = enc.fit(y_train)
        self.y = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels.csv")
        self.database_len = 590_326
        self.n_samples = self.database_len
        

    
    def __getitem__(self, index):
        self.X = plt.imread(f"C:\D\VS_programs_python\inzynierka\BigearthNet_png\{index}.png")
        return self.X, self.y.iloc[index-1].values[1:]
    
    def __len__(self):
        return self.n_samples