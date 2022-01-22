from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from torch import nn


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prune_by_mask(self, layer: nn.Module, mask: np.ndarray) -> None:
        pass

    @abstractmethod
    def prune_by_indicies(self, layer: nn.Module, indicies: List[int]) -> None:
        pass
