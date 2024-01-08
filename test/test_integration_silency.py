import pytest
from src.faithfulness.perturbation import *
from src.faithfulness.metrics import *
import torch
from src.EuroSat_dataloaders import *
from src.faithfulness.silency_map import *
from torchvision.transforms import v2
from torchvision.datasets import EuroSAT

resnet50_plus = torch.load("finished\\ResNet50_new\\resnet50_model.pth")
resnet50_plus.cuda()
resnet50_plus.eval()

# dataset:
transforms = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32),
    v2.Resize(224,antialias=None),
])
ds = EuroSAT("../EuroSat",transform=transforms,target_transform=transformation_eurosat,download=False)
ds_test = Test_Dataset_EuroSat(ds)

# target layer:
target_layers = [resnet50_plus.layer4[-1]]
targets = None

sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)

testdata = [(ds_test[np.random.randint(0,len(ds_test))][0].unsqueeze(0).cuda(),"grad_cam") for i in range(1,10)]

@pytest.mark.parametrize('sample,cam_type', testdata)
def test_sil_get_silency_map_(sample,cam_type):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map_(sample,cam_type = cam_type)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata2 = [(ds_test[np.random.randint(0,len(ds_test))][0].unsqueeze(0).cuda(),"") for i in range(1,10)]

@pytest.mark.parametrize('sample,cam_type', testdata2)
def test_sil_get_silency_map_plus(sample,cam_type):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map_(sample,cam_type = cam_type)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata3 = [(np.random.randint(0,len(ds_test)),"") for i in range(1,10)]

@pytest.mark.parametrize('sample,cam_type', testdata3)
def test_sil_get_silency_map_nr_plus(sample,cam_type):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map(sample,cam_type = cam_type)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata4 = [(np.random.randint(0,len(ds_test))) for i in range(1,10)]

@pytest.mark.parametrize('sample', testdata4)

def test_sil_get_silency_map_nr_(sample):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map(sample)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata5 = [(ds_test[np.random.randint(0,len(ds_test))][0],"grad_cam") for i in range(1,10)]

@pytest.mark.parametrize('sample,cam_type', testdata5)
def test_sil_get_silency_map_input_(sample,cam_type):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map_input(sample,cam_type = cam_type)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata6 = [(ds_test[np.random.randint(0,len(ds_test))][0],"") for i in range(1,10)]

@pytest.mark.parametrize('sample,cam_type', testdata6)
def test_sil_get_silency_map_input_plus(sample,cam_type):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map_input(sample,cam_type = cam_type)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata7 = [(ds_test[np.random.randint(0,len(ds_test))][0]) for i in range(1,10)]

@pytest.mark.parametrize('sample', testdata7)
def test_sil_get_silency_map_input(sample):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_silency_map_input(sample)
    assert type(temp) is np.ndarray
    assert temp.dtype.type == np.float32
    assert temp.shape == (224,224)

testdata8 = [(np.random.randint(0,len(ds_test)),) for i in range(1,10)]

@pytest.mark.parametrize('sample', testdata8)
def test_sil_get_pair_sailency(sample):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_pair_sailency(sample)
    assert len(temp) == 2
    assert type(temp[0]) is np.ndarray
    assert type(temp[1]) is np.ndarray
    assert temp[0].dtype.type == np.float32
    assert temp[0].shape == (224,224)
    assert temp[1].dtype.type == np.float32
    assert temp[1].shape == (224,224)

testdata9 = [(np.random.randint(0,len(ds_test)),True,False) for i in range(1,10)]

@pytest.mark.parametrize('sample,return_pred,return_perturbated_input', testdata9)
def test_sil_get_pair_sailency_2(sample,return_pred,return_perturbated_input):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_pair_sailency(sample,return_pred=return_pred,return_perturbated_input=return_perturbated_input)
    assert len(temp) == 3
    assert type(temp[0]) is np.ndarray
    assert type(temp[1]) is np.ndarray
    assert type(temp[2]) is np.int64
    assert temp[0].dtype.type == np.float32
    assert temp[0].shape == (224,224)
    assert temp[1].dtype.type == np.float32
    assert temp[1].shape == (224,224)

testdata9 = [(np.random.randint(0,len(ds_test)),False,True) for i in range(1,10)]

@pytest.mark.parametrize('sample,return_pred,return_perturbated_input', testdata9)
def test_sil_get_pair_sailency_2(sample,return_pred,return_perturbated_input):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_pair_sailency(sample,return_pred=return_pred,return_perturbated_input=return_perturbated_input)
    assert len(temp) == 3
    assert type(temp[0]) is np.ndarray
    assert type(temp[1]) is np.ndarray
    assert type(temp[2]) is torch.Tensor
    assert temp[2].device.type == "cuda"
    assert temp[0].dtype.type == np.float32
    assert temp[0].shape == (224,224)
    assert temp[1].dtype.type == np.float32
    assert temp[1].shape == (224,224)
    assert temp[2].dtype == torch.float32
    assert temp[2].shape == torch.Size((1,3,224,224))

testdata10 = [(np.random.randint(0,len(ds_test)),True,True) for i in range(1,10)]

@pytest.mark.parametrize('sample,return_pred,return_perturbated_input', testdata10)
def test_sil_get_pair_sailency_3(sample,return_pred,return_perturbated_input):
    ds_test = Test_Dataset_EuroSat(ds)

    # target layer:
    target_layers = [resnet50_plus.layer4[-1]]
    targets = None

    sil_gen = Silency_map_gen(resnet50_plus, ds_test, target_layers)
    temp = sil_gen.get_pair_sailency(sample,return_pred=return_pred,return_perturbated_input=return_perturbated_input)
    assert len(temp) == 4
    assert type(temp[0]) is np.ndarray
    assert type(temp[1]) is np.ndarray
    assert type(temp[2]) is np.int64
    assert type(temp[3]) is torch.Tensor
    assert temp[3].device.type == "cuda"
    assert temp[0].dtype.type == np.float32
    assert temp[0].shape == (224,224)
    assert temp[1].dtype.type == np.float32
    assert temp[1].shape == (224,224)
    assert temp[3].dtype == torch.float32
    assert temp[3].shape == torch.Size((1,3,224,224))