import pandas as pd
import torch
import tqdm

import matplotlib.pyplot as plt
import numpy as np
from src.faithfulness.perturbation import eurosat_perturbation, thr_fc, thr_fc_bin
from src.faithfulness.silency_map import Silency_map_gen
from src.faithfulness.metrics import *



class FaithfulnessMeasurment:
    def __init__(self,model, target_layers, ds, cam_type = "grad_cam"):
        self.model = model
        self.ds = ds
        self.cam_type = cam_type
        self.sil_gen = Silency_map_gen(model, ds, target_layers)

    def get_NSS_scores(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc, tr_fc_b = thr_fc_bin):
        sum_ = 0.0
        score_lst = []
        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            sl , pert_sl = self.sil_gen.get_pair_sailency(i,tr_fc=tr_fc,cam_type=self.cam_type,perturbation_func=perturbation_fc)
            score_ = NSS_func(sl , pert_sl, tr_fc = tr_fc_b)
            score_lst.append(score_)
            sum_ += score_
        return sum_/len(self.ds),score_lst

    def get_IG_scores(self, perturbation_fc=eurosat_perturbation, e=1, tr_fc = thr_fc, tr_fc_b = thr_fc_bin):
        sum_ = 0.0
        baseline_im = torch.zeros(self.sil_gen.ds[0][0].shape)
        baseline_sl = self.sil_gen.get_silency_map_input(baseline_im)
        score_lst = []
        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            sl , pert_sl = self.sil_gen.get_pair_sailency(i,tr_fc=tr_fc,cam_type=self.cam_type,perturbation_func=perturbation_fc)
            score_ = IG_func(sl , pert_sl, baseline_sl, e=e, tr_fc = tr_fc_b)
            score_lst.append(score_)
            sum_ += score_
        return sum_/len(self.ds),score_lst

    def get_MSE_scores(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc, tr_fc_b = thr_fc_bin):
        sum_ = 0.0
        score_lst = []
        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            score_ = MSE_func(self.sil_gen, i, self.model, perturbation_fc)
            score_lst.append(score_)
            sum_ += score_
        return sum_/len(self.ds),score_lst

    def get_SIM_scores(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc, no_bins = 20):
        sum_ = 0.0
        score_lst = []
        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            sl , pert_sl = self.sil_gen.get_pair_sailency(i,tr_fc=tr_fc,cam_type=self.cam_type,perturbation_func=perturbation_fc)
            score_ = SIM_func(sl , pert_sl, no_bins=no_bins)
            score_lst.append(score_)
            sum_ += score_
        return sum_/len(self.ds),score_lst

    def get_CC_scores(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc):
        sum_ = 0.0
        score_lst = []
        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            sl , pert_sl = self.sil_gen.get_pair_sailency(i,tr_fc=tr_fc,cam_type=self.cam_type,perturbation_func=perturbation_fc)
            score_ = CC_func(sl , pert_sl)
            score_lst.append(score_)
            sum_ += score_
        return sum_/len(self.ds),score_lst
    
    def get_all_scores(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc ):
        score = [0,0,0,0,0]
        lst_scores = [[],[],[],[],[]]
        score[0], lst_scores[0] = self.get_NSS_scores(perturbation_fc=perturbation_fc,tr_fc = tr_fc)
        data = pd.DataFrame(lst_scores[0], columns=["NSS"])
        data.to_csv("finished/ResNet50/faithfulness_metrics_NSS.csv",index=False)
        score[1], lst_scores[1] = self.get_IG_scores(perturbation_fc=perturbation_fc,tr_fc = tr_fc)
        data = pd.DataFrame(lst_scores[1], columns=["IG"])
        data.to_csv("finished/ResNet50/faithfulness_metrics_IG.csv",index=False)
        score[2], lst_scores[2] = self.get_MSE_scores(perturbation_fc=perturbation_fc,tr_fc = tr_fc)
        data = pd.DataFrame(lst_scores[2], columns=["MSE"])
        data.to_csv("finished/ResNet50/faithfulness_metrics_MSE.csv",index=False)
        score[3], lst_scores[3] = self.get_SIM_scores(perturbation_fc=perturbation_fc,tr_fc = tr_fc)
        data = pd.DataFrame(lst_scores[3], columns=["SIM"])
        data.to_csv("finished/ResNet50/faithfulness_metrics_SIM.csv",index=False)
        score[4], lst_scores[4] = self.get_CC_scores(perturbation_fc=perturbation_fc,tr_fc = tr_fc)
        data = pd.DataFrame(lst_scores[4], columns=["CC"])
        data.to_csv("finished/ResNet50/faithfulness_metrics_CC.csv",index=False)
        return pd.DataFrame(np.array(lst_scores).T, columns=["NSS","IG","MSE","SIM","CC"])
    
    def get_all_same_sl_map(self, perturbation_fc=eurosat_perturbation, tr_fc = thr_fc, no_bins = 20, tr_fc_b = thr_fc_bin, e=1):
        score_lst = [[],[],[],[],[]]

        baseline_im = torch.zeros(self.sil_gen.ds[0][0].shape)
        baseline_sl = self.sil_gen.get_silency_map_input(baseline_im)

        for i,tq in zip(range(len(self.ds)),tqdm.tqdm(range(len(self.ds)))):
            sl , pert_sl, pert_input = self.sil_gen.get_pair_sailency(i,tr_fc=tr_fc,cam_type=self.cam_type,perturbation_func=perturbation_fc,return_perturbated_input=True)
            score_ = NSS_func(sl , pert_sl, tr_fc = tr_fc_b)
            score_lst[0].append(score_)
            score_ = IG_func(sl , pert_sl, baseline_sl, e=e, tr_fc = tr_fc_b)
            score_lst[1].append(score_)
            score_ = MSE_func_mask(pert_input, self.sil_gen, i, self.model, perturbation_fc)
            score_lst[2].append(score_)
            score_ = SIM_func(sl , pert_sl, no_bins=no_bins)
            score_lst[3].append(score_)
            score_ = CC_func(sl , pert_sl)
            score_lst[4].append(score_)
        return pd.DataFrame(np.array(score_lst).T, columns=["NSS","IG","MSE","SIM","CC"])