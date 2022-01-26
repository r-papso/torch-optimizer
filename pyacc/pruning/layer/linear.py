from typing import Iterable

from pruning.layer.standard import StandardPruner
from torch import nn


class LinearPruner(StandardPruner):
    def __init__(self) -> None:
        super().__init__()

    def _after_mask_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        out_features = getattr(layer, "out_features") - len(channels)
        setattr(layer, "out_features", out_features)

    def _after_channel_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        in_features = getattr(layer, "in_features") - len(channels)
        setattr(layer, "in_features", in_features)
