from typing import Iterable

from pyacc.pruning.layer.standard import StandardPruner
from torch import nn


class LinearPruner(StandardPruner):
    def __init__(self) -> None:
        super().__init__()

    def _after_mask_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        setattr(layer, "out_features", len(channels))

    def _after_channel_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        setattr(layer, "in_features", len(channels))
