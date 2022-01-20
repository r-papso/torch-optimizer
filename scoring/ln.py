import numpy as np

from typing import List
from torch import nn

from scoring.abstract import Scoring


class LnScoring(Scoring):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def get_score(self, layer: nn.Module) -> np.ndarray:
        return np.abs(np.power(layer.weight.detach().numpy(), self.__n))

    def get_scores(self, layers: List[nn.Module]) -> List[np.ndarray]:
        return [self.get_score(layer) for layer in layers]
