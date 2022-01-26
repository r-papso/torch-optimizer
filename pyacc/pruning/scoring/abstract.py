import torch

from abc import ABC, abstractmethod
from torch import nn


class Scoring(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_score(self, layer: nn.Module, name: str) -> torch.Tensor:
        pass
