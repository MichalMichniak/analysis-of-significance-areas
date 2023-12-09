import numpy as np
import torch
from torchvision.transforms import Resize

def thr_fc(silency_map):
    """
    weighted mask of silency map
    """
    silency_map = (silency_map > np.mean(silency_map))*(silency_map-np.mean(silency_map))/(np.max(silency_map)-np.mean(silency_map))
    return silency_map

def thr_fc_bin(silency_map):
    """
    binaryzation of silency map
    """
    silency_map = silency_map > np.mean(silency_map)
    return silency_map 

def eurosat_perturbation(input,mask):
    # mask is a binary map where perturbate image

    # images from eurosat have almost same mean and std for R,G,B channels
    sigma = 0.08485537767410278
    mean = 0.37282294034957886

    # combined noise
    mul_noise = (1+np.random.normal(mean,2*sigma,size = input.shape)*mask)
    additive_noise = np.random.normal(mean,sigma,size = input.shape)
    return torch.mul(input,torch.from_numpy(mul_noise).float()) + (torch.from_numpy(additive_noise).float()*mask)