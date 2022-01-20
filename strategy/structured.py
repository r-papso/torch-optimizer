from typing import List
from torch import nn

from structs.layer_mask import LayerMask
from strategy.abstract import Strategy


class StructuredStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def get_mask(self, layer: nn.Module, fraction: float) -> LayerMask:
        # TODO: implement
        pass

    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[LayerMask]:
        # TODO: implement
        pass
