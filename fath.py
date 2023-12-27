from src.faithfulness.FaithfulnessMeasurment import FaithfulnessMeasurment
from src.EuroSat_dataloaders import transformation_eurosat
from torchvision.datasets import EuroSAT
from src.EuroSat_dataloaders import Test_Dataset_EuroSat
from torchvision.transforms import v2
import torch
from src.faithfulness.perturbation import thr_fc
from src.faithfulness.perturbation import eurosat_perturbation_inverted

def threshold_fc(silency_map):
    return thr_fc(silency_map,scale=0.5)

if __name__ == '__main__':
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

    # target layer:
    cam_type = "grad_cam"
    target_layers = [resnet50.layer4[-1]]
    fmeasure = FaithfulnessMeasurment(resnet50, target_layers, ds_test)
    
    data = fmeasure.get_all_same_sl_map(tr_fc=threshold_fc,perturbation_fc=eurosat_perturbation_inverted)
    data.to_csv("finished/ResNet50_new/faithfulness_metrics_grad_cam_inverted_noise_0_5.csv",index=False)