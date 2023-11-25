from src.faithfulness.perturbation import eurosat_perturbation, thr_fc_bin
import numpy as np
import torch

def NSS_func(sl_map, pert_sl_map, tr_fc = thr_fc_bin):
    sl_map_bin = tr_fc(sl_map)
    if(np.std(pert_sl_map) != 0):
        sl_map_norm = (sl_map-np.mean(pert_sl_map))/(2*np.std(pert_sl_map))
    else:
        sl_map_norm = (sl_map-np.mean(pert_sl_map))
    sum_of_pixel_path = 0.0
    count = 0
    for i in range(len(sl_map)):
        for j in range(len(sl_map)):
            if(sl_map_bin[i][j] == 1):
                sum_of_pixel_path += sl_map_norm[i][j]
                count += 1
    if(count == 0):
        return 0
    return sum_of_pixel_path/float(count)


def IG_func(sl_map, pert_sl_map, baseline_sl_map, e=1, tr_fc = thr_fc_bin):
    sl_map_bin = tr_fc(sl_map)
    count = 0
    sum_of_pixel = 0.0
    for i in range(len(sl_map)):
        for j in range(len(sl_map)):
            if(sl_map_bin[i][j] == 1):
                sum_of_pixel += np.log2(pert_sl_map[i][j] + e) - np.log2(baseline_sl_map[i][j] + e)
                count += 1
    return sum_of_pixel/float(count)


def MSE_func(sil_gen, nr, model):
    mask = sil_gen.get_silency_map(nr)
    input_tensor_pert = eurosat_perturbation(sil_gen.ds[nr][0],mask).unsqueeze(0).cuda()
    input_tensor = sil_gen.ds[nr][0].unsqueeze(0).cuda()
    y = model(input_tensor)
    y_pert = model(input_tensor_pert)
    return float(torch.sum(((y-y_pert))**2).cpu())