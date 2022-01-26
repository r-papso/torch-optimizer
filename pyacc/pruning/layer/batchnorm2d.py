from typing import List

import torch
from pruning.layer.abstract import LayerPruner
from torch import nn


class BatchNorm2dPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def prunable_by_mask(self, layer: nn.Module) -> bool:
        return False

    def prune_by_mask(self, layer: nn.Module) -> List[int]:
        raise ValueError(f"{BatchNorm2dPruner} -> Pruning by mask is not supported.")

    def prune_by_channels(self, layer: nn.Module, channels: List[int]) -> bool:
        assert isinstance(layer, nn.BatchNorm2d), f"Invalid layer type: {type(layer)}."

        layer.zero_grad()
        with torch.no_grad():
            slices = (channels,)
            param_names = ["running_mean", "running_var", "weight", "bias"]

            for name in param_names:
                self._prune_parameter(layer, name, slices)

        layer.num_features -= len(channels)
        return True
