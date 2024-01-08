import pytest
from src.faithfulness.perturbation import *
from src.faithfulness.metrics import *
import torch
from src.EuroSat_dataloaders import *
from src.faithfulness.silency_map import *
from torchvision.transforms import v2
from torchvision.datasets import EuroSAT
from src.faithfulness.FaithfulnessMeasurment import *

def threshold_fc(silency_map):
    return thr_fc(silency_map,scale=1)
# model load:
resnet50 = torch.load("finished\\ResNet50_new\\resnet50_model.pth")
resnet50.cuda()

# add softmax

fc_ = list(resnet50.fc)
fc_.append(torch.nn.Softmax(dim=1))
resnet50.fc = torch.nn.Sequential(*fc_)
resnet50.eval()

# dataset:
transforms = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32),
    v2.Resize(224,antialias=None),
])
ds = EuroSAT("../EuroSat",transform=transforms,target_transform=transformation_eurosat,download=False)
ds_test = Test_Dataset_EuroSat(ds)

class Ds_small(Dataset):
    def __init__(self,ds):
        self.tab = [1,2,3,4,5,6]
    
    def __len__(self):
        return len(self.tab)
    
    def __getitem__(self, index):
        return ds[self.tab[index]]

ds_small = Ds_small(ds_test)

# target layer:
cam_type = "grad_cam"
target_layers = [resnet50.layer4[-1]]
fmeasure = FaithfulnessMeasurment(resnet50, target_layers, ds_small)



def test_get_NSS_scores():
    assert type(fmeasure.get_NSS_scores()) == tuple
    assert len(fmeasure.get_NSS_scores()) == 2
    assert type(fmeasure.get_NSS_scores()[0]) == float
    assert type(fmeasure.get_NSS_scores()[1]) == list

def test_get_IG_scores():
    assert type(fmeasure.get_IG_scores()) == tuple
    assert len(fmeasure.get_IG_scores()) == 2
    assert type(fmeasure.get_IG_scores()[0]) == float
    assert type(fmeasure.get_IG_scores()[1]) == list

def test_get_MSE_scores():
    assert type(fmeasure.get_MSE_scores()) == tuple
    assert len(fmeasure.get_MSE_scores()) == 2
    assert type(fmeasure.get_MSE_scores()[0]) == float
    assert type(fmeasure.get_MSE_scores()[1]) == list

def test_get_SIM_scores():
    assert type(fmeasure.get_SIM_scores()) == tuple
    assert len(fmeasure.get_SIM_scores()) == 2
    assert type(fmeasure.get_SIM_scores()[0]) == float
    assert type(fmeasure.get_SIM_scores()[1]) == list

def test_get_CC_scores():
    assert type(fmeasure.get_CC_scores()) == tuple
    assert len(fmeasure.get_CC_scores()) == 2
    assert type(fmeasure.get_CC_scores()[0]) == float
    assert type(fmeasure.get_CC_scores()[1]) == list

def test_get_CC_scores():
    assert type(fmeasure.get_CC_scores()) == tuple
    assert len(fmeasure.get_CC_scores()) == 2
    assert type(fmeasure.get_CC_scores()[0]) == float
    assert type(fmeasure.get_CC_scores()[1]) == list

def test_get_all_same_sl_map():
    assert type(fmeasure.get_all_same_sl_map()) == pd.DataFrame
