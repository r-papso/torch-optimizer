from ast import Tuple
import numpy as np
from torch import nn
from pruning.layer.abstract import LayerPruner


class BatchNorm2dPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module, mask: np.ndarray) -> Tuple[int]:
        raise ValueError(f"{BatchNorm2dPruner}: pruning by mask is not supported")

    def prune_by_input(self, layer: nn.Module, input_shape: Tuple[int]) -> Tuple[int]:
        # TODO implement
        pass
