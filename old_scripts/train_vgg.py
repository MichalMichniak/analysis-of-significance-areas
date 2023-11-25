from src.VGG16_model_BE import VGG16_model_transfer
from src.dataset_info import *
import torch
from src.Bigearth import Bigearth_Pruned, Train_Dataset, Test_Dataset
from torch.utils.data import Dataset,DataLoader
import datetime

if __name__ == "__main__":
    vgg_16 = VGG16_model_transfer()
    vgg_16.load(13,True,conv_layers_train=True)

    bigearth_dl = Bigearth_Pruned()
    bigearth_dl = Bigearth_Pruned()
    dataset = Train_Dataset(bigearth_dl,batch_len=8)
    test_dataset = Test_Dataset(bigearth_dl,batch_len=8)
    dataloader = DataLoader(dataset=dataset, batch_size=16,shuffle= True,num_workers=2 )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16,shuffle= True,num_workers=2 )
    train_loss,test_loss,train_accuracy,test_accuracy = vgg_16.train(26,dataloader,test_dataloader)
    