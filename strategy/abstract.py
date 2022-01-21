from abc import ABC, abstractmethod
from typing import List
import numpy as np
from torch import nn

from scoring.abstract import Scoring


class Strategy(ABC):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__()

        self._scoring = scoring

    @abstractmethod
    def get_mask(self, layer: nn.Module, fraction: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[np.ndarray]:
        pass
