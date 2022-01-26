import torch
from torch import nn

from scoring.abstract import Scoring


class LnScoring(Scoring):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def get_score(self, layer: nn.Module, name: str) -> torch.Tensor:
        param = getattr(layer, name)
        return torch.abs(torch.float_power(param, self.__n))
