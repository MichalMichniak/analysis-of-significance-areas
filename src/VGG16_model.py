import torch
import torchvision

class VGG16_model_transfer:
    """
    VGG16 architecture without cuda
    from transfer learning
    """
    def __init__(self) -> None:
        pass

    def load_new(self, n_labels) -> None:
        new_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=n_labels, bias=True)
        )
        self.vgg16 = torchvision.models.vgg16_bn(pretrained=True, progress=True)
        for layer in self.vgg16.features.parameters():
            layer.require_grad = False
        self.vgg16.classifier = new_classifier

    def forward_pass(self, x : torch.Tensor):
        return self.vgg16(x)