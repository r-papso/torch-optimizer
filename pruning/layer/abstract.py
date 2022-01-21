from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from torch import nn


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prune_by_mask(self, layer: nn.Module, mask: np.ndarray) -> Tuple[int]:
        pass

    @abstractmethod
    def prune_by_input(self, layer: nn.Module, input_shape: Tuple[int]) -> Tuple[int]:
        pass

    @abstractmethod
    def validate_input(self, layer: nn.Module, input_shape: Tuple[int]) -> bool:
        pass
