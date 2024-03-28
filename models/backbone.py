from email.policy import strict
import numbers
import torch.nn as nn
import torchvision.models as models
from timm.models import create_model

class BaseException(Exception):
    """Base exception"""
class InvalidDatasetSelection(BaseException):
    """Raised when the choice of dataset is invalid."""
class InvalidBackboneError(BaseException):
    """Raised when the choice of backbone Convnet is invalid."""

check_dict = {"tiny": "/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_tiny_22k_224.pth",
            "small":"/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_small_22k_224.pth",
            "base":"/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_tiny_22k_224.pth"}

class BACKBONE_ABiD(nn.Module):
    def __init__(self, base_model,out_dim):
        super(BACKBONE_ABiD, self).__init__()
        self.out_dim = out_dim
        if "resnet" in  base_model:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                                "resnet50": models.resnet50(pretrained=True),
                                "resnet101": models.resnet101(pretrained=True),
            }
            self.backbone = self._get_basemodel(base_model)
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            self.resnet_dict1 = {"ConvneXt_small":create_model("convnext_small", pretrained=True,checkpoint_path='/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_small_22k_224.pth',strict=False,num_classes=self.out_dim),
                                "ConvneXt_tiny":create_model("convnext_tiny",pretrained=True,checkpoint_path='/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_tiny_22k_224.pth',strict=False,num_classes=self.out_dim),
                                "ConvneXt_base":create_model("convnext_base",pretrained=True,checkpoint_path='/home/mayuan1/FB-code/cuda4/cuda1/params/convnext_tiny_22k_224.pth',strict=False,num_classes=self.out_dim),}
            self.backbone = self._get_basemodel(base_model)

    def _get_basemodel(self, model_name):
        try:
            if "ConvneXt" in model_name:
                model = self.resnet_dict1[model_name]
            else:
                model = self.resnet_dict[model_name]
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs,self.out_dim)
        except KeyError:
            raise InvalidBackboneError("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50 or resnet101")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
