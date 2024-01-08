from src.faithfulness.FaithfulnessMeasurment import FaithfulnessMeasurment
from src.EuroSat_dataloaders import transformation_eurosat
from torchvision.datasets import EuroSAT
from src.EuroSat_dataloaders import Test_Dataset_EuroSat
from torchvision.transforms import v2
import torch
from src.faithfulness.perturbation import thr_fc
from src.faithfulness.perturbation import eurosat_perturbation_inverted


if __name__ == '__main__':
    for noise,name in zip([0.5,1.0,1.5],["0_5","1","1_5"]):
        def threshold_fc(silency_map):
            return thr_fc(silency_map,scale=noise)
        # model load:
        vgg = torch.load("finished\\VGG16\\vgg_model.pth")
        vgg.cuda()
        for param in vgg.features.parameters():
                param.requires_grad = True
        # add softmax
        fc_ = list(vgg.classifier)
        fc_.append(torch.nn.Softmax(dim=1))
        vgg.classifier = torch.nn.Sequential(*fc_)
        vgg.eval()

        # model for grad cam ++
        vgg_plus = torch.load("finished\\VGG16\\vgg_model.pth")
        vgg_plus.cuda()
        for param in vgg_plus.features.parameters():
                param.requires_grad = True
        
        # dataset:
        transforms = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
            v2.Resize(224,antialias=None),
        ])
        ds = EuroSAT("../EuroSat",transform=transforms,target_transform=transformation_eurosat,download=False)
        ds_test = Test_Dataset_EuroSat(ds)

        # target layer:
        cam_type = "grad_cam_plus_plus"
        target_layers = [vgg_plus.features[-2]]
        fmeasure = FaithfulnessMeasurment(vgg, target_layers, ds_test,cam_type=cam_type,model_grad_cam_plus_plus=vgg_plus)
        
        data = fmeasure.get_all_same_sl_map(tr_fc=threshold_fc,perturbation_fc=eurosat_perturbation_inverted)
        data.to_csv(f"finished/VGG16/faithfulness_metrics_grad_cam_plus_plus_inverted_noise_{name}.csv",index=False)