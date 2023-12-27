import pytest
from src.faithfulness.perturbation import *
from src.faithfulness.metrics import *

testdata = [np.random.random((10,10)) for i in range(100)]

@pytest.mark.parametrize('sample', testdata)
def test_thr_fc(sample):
    assert np.min(thr_fc(sample)) ==0
    assert np.max(thr_fc(sample)) ==1

testdata2 = [(np.array([[0,0],[0,1]]),(([1,1],1),)), 
             (np.array([[0,1],[0,1]]),(([1,1],1),([0,1],1))),
             (np.array([[0,0.5,0.5],[0,0,0],[0,1,0]]),(([2,1],1),([1,1],0)))]


@pytest.mark.parametrize('sample,ones_t', testdata2)
def test_thr_fc2(sample,ones_t):
    for i in ones_t:
        assert thr_fc(sample)[i[0][0],i[0][1]] == i[1]
    assert np.min(thr_fc(sample)) == 0

testdata2 = [(np.array([[0,0],[0,1]]),(([1,1],1),)), 
             (np.array([[0,1],[0,1]]),(([1,1],1),([0,1],1))),
             (np.array([[0,0.5,0.5],[0,0,0],[0,1,0]]),(([2,1],1),([0,1],1),([0,2],1))),
             (np.array([[1,1.5,1.5],[1,1,1],[1,2,1]]),(([2,1],1),([0,1],1),([0,2],1),([0,1],1),([0,0],0))),]


@pytest.mark.parametrize('sample,ones_t', testdata2)
def test_thr_fc_bin(sample,ones_t):
    for i in ones_t:
        assert thr_fc_bin(sample)[i[0][0],i[0][1]] == i[1]
    assert np.min(thr_fc_bin(sample)) == 0

import torch
test_pert_list = [(torch.tensor([[[1,1],[1,1]] for i in range(3)]),np.array([[1,0],[0,1]]),(3,2,2)),
                  (torch.tensor([[[1,1,1],[1,1,1],[1,1,1]] for i in range(3)]),np.array([[1,0,0],[0.5,0.5,0],[0,0,1]]),(3,3,3))]

@pytest.mark.parametrize('input,mask,shape_t', test_pert_list)
def test_eurosat_perturbation(input,mask,shape_t):
    #shape and type test
    assert eurosat_perturbation(input,mask).shape == shape_t
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] == 0:
                assert eurosat_perturbation(input,mask)[0,i,j] == input[0,i,j]
    assert type(eurosat_perturbation(input,mask)) == torch.Tensor


test_NSS_list = [
    (np.array([[0,1],[0,1]]),np.array([[0,0],[0,1]]), thr_fc_bin,1.15470053/2),
    (q:=np.array([[0,1],[0,1]]),p:=np.array([[0,0],[1,1]]), thr_fc_bin,((t:=(p-np.mean(p))/np.std(p))[0,1]+t[1,1])/2),
    (q:=np.array([[0,0,0],[0,0,0],[0,0,1]]),p:=np.array([[0,0,0],[1,1,1],[0,0,1]]), thr_fc_bin,((t:=(p-np.mean(p))/np.std(p))[2,2])),
    (q:=np.array([[0,1,0],[0,0,0],[0,0,1]]),p:=np.array([[0,0,0],[1,1,1],[0,1,1]]), thr_fc_bin,((t:=(p-np.mean(p))/np.std(p))[0,1]+t[2,2])/2),
    (q:=np.zeros([10,10]),p:=np.ones([10,10]),thr_fc,0),
    (q:=np.ones([10,10]),p:=np.ones([10,10]),thr_fc,0)
]

@pytest.mark.parametrize('sl_map, pert_sl_map, tr_fc, result', test_NSS_list)
def test_NSS_func(sl_map, pert_sl_map, tr_fc, result):
    eps= 1e-6
    assert NSS_func(sl_map, pert_sl_map, tr_fc) > result - eps
    assert NSS_func(sl_map, pert_sl_map, tr_fc) < result + eps

test_IG_list = [
    (q:=np.array([[0,1],[0,1]]),p:=np.array([[0,0],[1,1]]),b:=np.array([[0,0],[0,0]]),np.sum(np.log2(1+p)*q)/2,thr_fc_bin),
    (q:=np.array([[0,0,0],[0,0,0],[0,0,1]]),p:=np.array([[0,0,0],[1,1,1],[0,0,1]]),b:=np.array([[0,0,0],[0,0,0],[0,0,0]]),np.sum(np.log2(1+p)*q),thr_fc_bin),
    (q:=np.array([[0,1,0],[0,0,0],[0,0,1]]),p:=np.array([[0,0,0],[1,1,1],[0,1,1]]),b:=np.array([[0,0,1],[0,0,0.2],[0,0,0]]),np.sum((np.log2(1+p)-np.log2(1+b))*q)/2,thr_fc_bin),
    (q:=np.zeros([10,10]),p:=np.ones([10,10]), b:=np.zeros([10,10]),0,thr_fc_bin),
    (q:=np.ones([10,10]),p:=np.ones([10,10]), b:=np.zeros([10,10]),0,thr_fc_bin),
    (q:=np.ones([10,10]),p:=np.ones([10,10]), b:=np.ones([10,10])/2,np.sum((np.log2(1+p)-np.log2(1+b))*q)/100,lambda x: np.ones(x.shape))
]

@pytest.mark.parametrize('sl_map, pert_sl_map, baseline_sl_map, result, tr_fc', test_IG_list)
def test_IG_func(sl_map, pert_sl_map, baseline_sl_map, result, tr_fc, e=1):
    eps= 1e-6
    assert IG_func(sl_map, pert_sl_map, baseline_sl_map, e, tr_fc) > result - eps
    assert IG_func(sl_map, pert_sl_map, baseline_sl_map, e, tr_fc) < result + eps

test_SIM_list = [(np.random.uniform(0,1,[20,20]),np.random.uniform(0,1,[20,20])) for i in range(10)]
@pytest.mark.parametrize('sl_map, pert_sl_map', test_SIM_list)
def test_SIM_func(sl_map, pert_sl_map):
    sl_map[5,5] = 1
    sl_map[4,4] = 0
    pert_sl_map[5,2] = 1
    pert_sl_map[3,4] = 0
    # in test both histograms are from (0,1) and reach both 0 and 1
    first_hist,_  = np.histogram(sl_map,bins=20)
    first_hist =first_hist/np.sum(first_hist)
    secound_hist, _ = np.histogram(pert_sl_map,bins=20)/np.sum(np.histogram(pert_sl_map)[1])
    secound_hist =secound_hist/np.sum(secound_hist)
    # elementwise min
    minimum = np.minimum(first_hist,secound_hist)
    result = np.sum(minimum)
    eps= 1e-6
    assert SIM_func(sl_map, pert_sl_map, no_bins = 20, show = False) > result - eps
    assert SIM_func(sl_map, pert_sl_map, no_bins = 20, show = False) < result + eps
    pass

test_SIM_list2 = [(np.ones([20,20]),np.zeros([20,20]),0), (np.zeros([20,20]),np.zeros([20,20]),1)]
@pytest.mark.parametrize('sl_map, pert_sl_map,result', test_SIM_list2)
def test_SIM_func(sl_map, pert_sl_map,result):

    assert SIM_func(sl_map, pert_sl_map, no_bins = 20, show = False) == result
    pass


test_CC_list = [(np.random.uniform(0,1,[20,20]),np.random.uniform(0,1,[20,20])) for i in range(10)]
@pytest.mark.parametrize('sl_map, pert_sl_map', test_CC_list)
def test_CC_func(sl_map, pert_sl_map):
    result = np.cov(np.array([sl_map.flatten(),pert_sl_map.flatten()]))[0,1]/(sl_map.std() * pert_sl_map.std())
    eps= 1e-6
    assert CC_func(sl_map, pert_sl_map, no_bins = 20, show = False) > result - eps
    assert CC_func(sl_map, pert_sl_map, no_bins = 20, show = False) < result + eps
    assert -1<=CC_func(sl_map, pert_sl_map, no_bins = 20, show = False)<=1
    pass

test_CC_list2 = [(np.zeros([20]),np.zeros([20]),0),(np.ones([20]),np.ones([20]),1),(np.zeros([20]),np.ones([20]),0)]
@pytest.mark.parametrize('sl_map, pert_sl_map, result', test_CC_list2)
def test_CC_func2(sl_map, pert_sl_map,result):
    eps= 1e-6
    assert CC_func(sl_map, pert_sl_map, no_bins = 20, show = False) > result - eps
    assert CC_func(sl_map, pert_sl_map, no_bins = 20, show = False) < result + eps
    pass



