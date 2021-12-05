import numpy as np

from typing import List
from torch import nn

from pipeline.context import Context
from scoring.scorer import Scorer


class LnScorer(Scorer):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def get_score(self, layers: List[nn.Module], context: Context) -> List[np.ndarray]:
        return [np.abs(np.power(layer.weight.detach().numpy(), self.__n)) for layer in layers]
