from typing import Any

import torch
import torchvision.models.vgg
from torch import nn


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


torchvision.models.vgg.VGG = VGG


def vgg11(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg11(pretrained, progress, **kwargs)


def vgg11_bn(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg11_bn(pretrained, progress, **kwargs)


def vgg13(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg13(pretrained, progress, **kwargs)


def vgg13_bn(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg13_bn(pretrained, progress, **kwargs)


def vgg16(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg16(pretrained, progress, **kwargs)


def vgg16_bn(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg16_bn(pretrained, progress, **kwargs)


def vgg19(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg19(pretrained, progress, **kwargs)


def vgg19_bn(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> torchvision.models.vgg.VGG:
    return torchvision.models.vgg.vgg19_bn(pretrained, progress, **kwargs)
