from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.faithfulness.perturbation import eurosat_perturbation, thr_fc
import numpy as np
import torch

class Silency_map_gen:
    def __init__(self,model, dataset, target_layers):
        self.model = model
        self.ds = dataset
        self.target_layers = target_layers
        self.grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        self.grad_cam_plus_plus = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
        pass
    
    def get_silency_map_(self,input_tensor ,targets = None, cam_type = "grad_cam"):
        silency_map = None
        if cam_type == "grad_cam":
            if targets == None:
                silency_map = self.grad_cam(input_tensor=input_tensor, targets=targets)
            else:
                with GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True) as cam:
                    silency_map = cam(input_tensor=input_tensor, targets=targets)
        else:
            if targets == None:
                silency_map = self.grad_cam_plus_plus(input_tensor=input_tensor, targets=targets)
            else:
                with GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=True) as cam:
                    silency_map = cam(input_tensor=input_tensor, targets=targets)
        return silency_map[0, :]


    def get_silency_map(self,nr,targets = None, cam_type = "grad_cam"):
        input_tensor = self.ds[nr][0].unsqueeze(0).cuda()
        return self.get_silency_map_(input_tensor,targets, cam_type)
    
    def get_silency_map_input(self,input_tensor,targets = None, cam_type = "grad_cam"):
        input_tensor = input_tensor.unsqueeze(0).cuda()
        return self.get_silency_map_(input_tensor,targets, cam_type)

    def get_perturbated_silency_map(self,nr,mask = None,targets = None, cam_type = "grad_cam", perturbation_func = eurosat_perturbation, tr_fc = thr_fc, return_perturbated_input = False):
        if mask is None:
            mask = self.get_silency_map(nr,targets, cam_type)
            mask = tr_fc(mask)
        input_tensor = eurosat_perturbation(self.ds[nr][0],mask).unsqueeze(0).cuda()
        if return_perturbated_input:
            return self.get_silency_map_(input_tensor,targets, cam_type),input_tensor
        return self.get_silency_map_(input_tensor,targets, cam_type)

    def get_pair_sailency(self,nr, tr_fc = thr_fc,targets = None, cam_type = "grad_cam", perturbation_func = eurosat_perturbation, return_pred = False, return_perturbated_input = False):
        input_tensor = self.ds[nr][0].unsqueeze(0).cuda()
        ground_truth_map = self.get_silency_map_(input_tensor,targets, cam_type)
        if targets is None:
            pred = self.model(input_tensor)
            targets = [ClassifierOutputTarget(np.argmax(pred[0].cpu().detach().numpy()))]

        mask = tr_fc(ground_truth_map)
        if return_perturbated_input:
            perturbated_map, pert_input = self.get_perturbated_silency_map(nr ,mask ,targets, cam_type, perturbation_func, tr_fc,return_perturbated_input=return_perturbated_input)
        else:
            perturbated_map = self.get_perturbated_silency_map(nr ,mask ,targets, cam_type, perturbation_func, tr_fc,return_perturbated_input=return_perturbated_input)
        if return_pred:
            if return_perturbated_input:
                return ground_truth_map, perturbated_map,np.argmax(pred.cpu().detach().numpy()), pert_input
            return ground_truth_map, perturbated_map,np.argmax(pred.cpu().detach().numpy())
        if return_perturbated_input:
            return ground_truth_map, perturbated_map, pert_input
        return ground_truth_map, perturbated_map
    
