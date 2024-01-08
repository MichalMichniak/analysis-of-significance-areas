import pytest
from src.faithfulness.perturbation import *
from src.faithfulness.metrics import *
import torch

seed_nr = 1234

testdata = [np.random.random((i,i)) for i in range(1,100)]

@pytest.mark.parametrize('sample', testdata)
def test_thr_fc(sample):
    assert type(thr_fc(sample)) is np.ndarray
    assert thr_fc(sample).dtype.type is np.float64
    assert thr_fc(sample).shape == sample.shape

testdata2 = [np.random.random((i,i)).astype(np.float64) for i in range(1,100)]

@pytest.mark.parametrize('sample', testdata2)
def test_thr_fc_bin(sample):
    assert type(thr_fc_bin(sample)) is np.ndarray
    assert thr_fc_bin(sample).dtype.type is np.float64
    assert thr_fc_bin(sample).shape == sample.shape

test_pert_list = [(torch.from_numpy(np.random.uniform(size=(3,i,i))),np.random.uniform(0,1,size=(i,i))) for i in range(100)]

@pytest.mark.parametrize('input,mask', test_pert_list)
def test_eurosat_perturbation(input,mask):
    assert type(eurosat_perturbation(input,mask)) is torch.Tensor
    assert eurosat_perturbation(input,mask).dtype is torch.float64
    assert eurosat_perturbation(input,mask).shape == input.shape

test_pert_list = [(torch.from_numpy(np.random.uniform(size=(3,i,i))),np.random.uniform(0,1,size=(i,i))) for i in range(100)]

@pytest.mark.parametrize('input,mask', test_pert_list)
def test_eurosat_perturbation_inverted(input,mask):
    assert type(eurosat_perturbation_inverted(input,mask)) is torch.Tensor
    assert eurosat_perturbation_inverted(input,mask).dtype is torch.float64
    assert eurosat_perturbation_inverted(input,mask).shape == input.shape


test_NSS_list = [(np.random.uniform(size=(i,i)),np.random.uniform(size=(i,i)),thr_fc_bin) for i in range(1,10)]

@pytest.mark.parametrize('sl_map, pert_sl_map, tr_fc', test_NSS_list)
def test_NSS_func(sl_map, pert_sl_map, tr_fc):
    assert type(NSS_func(sl_map, pert_sl_map, tr_fc)) == float

test_IG_list = [
    (np.random.uniform(size=(i,i)),np.random.uniform(size=(i,i)),np.random.uniform(size=(i,i)),thr_fc_bin) for i in range(100)
]

@pytest.mark.parametrize('sl_map, pert_sl_map, baseline_sl_map, tr_fc', test_IG_list)
def test_IG_func(sl_map, pert_sl_map, baseline_sl_map, tr_fc, e=1):
    assert type(IG_func(sl_map, pert_sl_map, baseline_sl_map, e, tr_fc)) == float

test_SIM_list = [(np.random.uniform(0,1,[i,i]),np.random.uniform(0,1,[i,i])) for i in range(1,10)]
@pytest.mark.parametrize('sl_map, pert_sl_map', test_SIM_list)
def test_SIM_func(sl_map, pert_sl_map):
    assert type(SIM_func(sl_map, pert_sl_map, no_bins = 20, show = False)) is float

test_CC_list = [(np.random.uniform(0,1,[i,i]),np.random.uniform(0,1,[i,i])) for i in range(1,10)]
@pytest.mark.parametrize('sl_map, pert_sl_map', test_CC_list)
def test_CC_func(sl_map, pert_sl_map):
    assert type(CC_func(sl_map, pert_sl_map)) is float



