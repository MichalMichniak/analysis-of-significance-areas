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
             (np.array([[0,0.5,0.5],[0,0,0],[0,1,0]]),(([2,1],1)))]


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
test_pert_list = [(torch.tensor([[[1,1],[1,1]]]),np.array([[1,0],[0,1]]),(1,2,2)),
                  (torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]]),np.array([[1,0,0],[0.5,0.5,0],[0,0,1]]),(1,3,3))]

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
    eps= 1e-3
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
    eps= 1e-3
    assert IG_func(sl_map, pert_sl_map, baseline_sl_map, e, tr_fc) > result - eps
    assert IG_func(sl_map, pert_sl_map, baseline_sl_map, e, tr_fc) < result + eps





