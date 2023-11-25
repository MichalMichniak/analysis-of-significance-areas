import numpy as np
import torch

def thr_fc(silency_map):
    """
    binaryzation of silency map
    """
    silency_map = silency_map>0.1
    return silency_map
    

def eurosat_perturbation(input,mask):
    sigma = 1
    mean = 0
    # combined noise
    mul_noise = (1+np.random.normal(mean,2*sigma,size = input.shape))
    additive_noise = np.random.normal(mean,sigma,size = input.shape)
    return torch.mul(input,torch.from_numpy(mul_noise).float()) + torch.from_numpy(additive_noise).float()