import torch
from src.EuroSat_dataloaders import transformation_eurosat

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader
import torchvision

from src.EuroSat_dataloaders import Train_Dataset_EuroSat,Test_Dataset_EuroSat
from src.ResNet50_model import ResNet50_model_transfer
import datetime

from src.EuroSat_dataloaders import Train_Dataset_EuroSat,Test_Dataset_EuroSat,Validation_Dataset_EuroSat


import warnings
warnings.filterwarnings("ignore")
torchvision.disable_beta_transforms_warning()

if __name__ == '__main__':
    torchvision.disable_beta_transforms_warning()
    transforms = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        v2.Resize(224),
    ])
    ds = EuroSAT("../EuroSat",transform=transforms,target_transform=transformation_eurosat,download=False)
    ds_train = Train_Dataset_EuroSat(ds)
    ds_test = Validation_Dataset_EuroSat(ds)
    train_dataloader = DataLoader(dataset=ds_train, batch_size=32,shuffle=True,num_workers=3)
    test_dataloader = DataLoader(dataset=ds_test, batch_size=32,shuffle=True,num_workers=3)
    resnet50 = ResNet50_model_transfer()
    resnet50.load(10,False,conv_layers_train=True)
    train_loss,test_loss,train_accuracy,test_accuracy = resnet50.train(36,train_dataloader,test_dataloader)
    print(train_loss,test_loss,train_accuracy,test_accuracy)
    date = f"{datetime.datetime.now()}"
    date = "_".join(date.split())
    date = "_".join(date.split(":"))
    date = "_".join(date.split("."))
    with open(f"logs/resnet_eurosat_{date}.csv","w") as file:
        for x,y,z,w in zip(train_loss,test_loss,train_accuracy,test_accuracy):
            file.write(f"{x},{y},{z},{w}\n")