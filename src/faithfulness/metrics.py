from src.faithfulness.perturbation import eurosat_perturbation, thr_fc_bin
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def MSE_func(sil_gen, nr, model, perturbation_fc = eurosat_perturbation):
    mask = sil_gen.get_silency_map(nr)
    input_tensor_pert = perturbation_fc(sil_gen.ds[nr][0],mask).unsqueeze(0).cuda()
    input_tensor = sil_gen.ds[nr][0].unsqueeze(0).cuda()
    y = model(input_tensor)
    y_pert = model(input_tensor_pert)
    return float(torch.sum(((y-y_pert))**2).cpu())

def MSE_func_mask(pert_input, sil_gen, nr, model, perturbation_fc = eurosat_perturbation):
    input_tensor_pert = pert_input
    input_tensor = sil_gen.ds[nr][0].unsqueeze(0).cuda()
    y = model(input_tensor)
    y_pert = model(input_tensor_pert)
    return float(torch.sum(((y-y_pert))**2).cpu())

def SIM_func(sl_map, pert_sl_map, no_bins = 20, show = False):
    upper = np.max([np.max(sl_map),np.max(pert_sl_map)])
    hist_sl = torch.histogram(torch.from_numpy(sl_map),no_bins,range=(0.0,upper),density = True)
    hist_pert_sl = torch.histogram(torch.from_numpy(pert_sl_map),no_bins,range=(0.0,upper),density = True)
    hist_sl_np = hist_sl.hist.numpy()*hist_sl.bin_edges.numpy()[1]
    hist_pert_sl_np = hist_pert_sl.hist.numpy()*hist_pert_sl.bin_edges.numpy()[1]
    if show:
        plt.plot(hist_sl.bin_edges.numpy()[1:],hist_sl.hist.numpy())
        plt.show()
        plt.plot(hist_pert_sl.bin_edges.numpy()[1:],hist_pert_sl.hist.numpy())
        plt.show()
    return np.sum(np.minimum(hist_sl_np,hist_pert_sl_np))

def CC_func(sl_map, pert_sl_map, no_bins = 20, show = False):
    """
    args:
        sl_map : np.array - silency map of target instance
        pert_sl_map  : np.array - perturbated silency map of target instance
        no_bins : int - number of elements in discrete probability distribution
        show : bool - show histograms
    """
    upper = np.max([np.max(sl_map),np.max(pert_sl_map)])
    if upper == 0:
        return 1
    hist_sl = torch.histogram(torch.from_numpy(sl_map),no_bins,range=(0.0,upper),density = True)
    hist_pert_sl = torch.histogram(torch.from_numpy(pert_sl_map),no_bins,range=(0.0,upper),density = True)
    hist_sl_np = hist_sl.hist.numpy()*hist_sl.bin_edges.numpy()[1]
    hist_pert_sl_np = hist_pert_sl.hist.numpy()*hist_pert_sl.bin_edges.numpy()[1]
    if show:
        plt.plot(hist_sl.bin_edges.numpy()[1:],hist_sl.hist.numpy())
        plt.show()
        plt.plot(hist_pert_sl.bin_edges.numpy()[1:],hist_pert_sl.hist.numpy())
        plt.show()
    temp = np.sqrt(np.mean((hist_sl_np*np.linspace(0,1,no_bins)-np.mean(hist_sl_np*np.linspace(0,1,no_bins)))**2)*np.mean((hist_pert_sl_np*np.linspace(0,1,no_bins)-np.mean(hist_pert_sl_np*np.linspace(0,1,no_bins)))**2))
    if temp == 0:
        if (np.mean(hist_sl_np) == np.mean(hist_pert_sl_np)) and (np.std(hist_sl_np) == np.std(hist_pert_sl_np)):
            return 1
        return 0
    cc = np.mean((hist_sl_np*np.linspace(0,1,no_bins)-np.mean(hist_sl_np*np.linspace(0,1,no_bins)))*(hist_pert_sl_np*np.linspace(0,1,no_bins)-np.mean(hist_pert_sl_np*np.linspace(0,1,no_bins))))/(temp)
    if np.isnan(cc):
        return 0
    return cc

