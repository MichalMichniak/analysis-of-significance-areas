import numpy as np
import torch
from torchvision.transforms import Resize

def thr_fc(silency_map, scale=1):
    """
    weighted mask of silency map
    """
    silency_map = (silency_map > np.mean(silency_map))*np.minimum(((silency_map-np.mean(silency_map))*scale/(np.max(silency_map)-np.mean(silency_map)))*2,scale)
    return silency_map

def thr_fc_bin(silency_map):
    """
    binaryzation of silency map
    """
    silency_map = silency_map > np.mean(silency_map)
    return silency_map.astype(np.float64)

def eurosat_perturbation(input,mask):
    # mask is a binary map where perturbate image

    # images from eurosat have almost same mean and std for R,G,B channels
    # sigma = 0.08485537767410278
    # mean = 0.37282294034957886
    mean = [-0.31124905 , -0.23941825 , -0.18445934]
    sigma = [0.1768554 , 0.12416639 , 0.1041581]
    # combined noise
    mul_noise = np.zeros(input.shape)
    additive_noise = np.zeros(input.shape)
    for i in range(3):
        mul_noise[i] = (1+np.random.normal(0,2*sigma[i],size = input[0].shape)*mask)
        additive_noise[i] = np.random.normal(0,sigma[i],size = input[0].shape)
    return torch.mul(input,torch.from_numpy(mul_noise).float()) + (torch.from_numpy(additive_noise).float()*mask)

def eurosat_perturbation_inverted(input,mask):
    # mask is a binary map where perturbate image

    # images from eurosat have almost same mean and std for R,G,B channels
    mean = [-0.31124905 , -0.23941825 , -0.18445934]
    sigma = [0.1768554 , 0.12416639 , 0.1041581]
    # inverted noise
    mul_noise = np.zeros(input.shape)
    additive_noise = np.zeros(input.shape)
    for i in range(3):
        mul_noise[i] = (1+(np.random.normal(0,2*sigma[i],size = input[0].shape)*mask))
        additive_noise[i] = np.random.normal(0,sigma[i],size = input[0].shape)
    return 1-torch.mul(1-input,torch.from_numpy(mul_noise).float()) + (torch.from_numpy(additive_noise).float()*mask)