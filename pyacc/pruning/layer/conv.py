from typing import Iterable

from pruning.layer.standard import StandardPruner
from torch import nn


class ConvPruner(StandardPruner):
    def __init__(self) -> None:
        super().__init__()

    def _after_mask_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        out_channels = getattr(layer, "out_channels") - len(channels)
        setattr(layer, "out_channels", out_channels)

    def _after_channel_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        in_channels = getattr(layer, "in_channels") - len(channels)
        setattr(layer, "in_channels", in_channels)
