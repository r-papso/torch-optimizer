import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, features: nn.Module, flatten: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.flatten = flatten
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
