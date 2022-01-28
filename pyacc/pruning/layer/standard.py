from typing import Iterable

import torch
from pyacc.pruning.layer.abstract import LayerPruner
from torch import nn


class StandardPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def prunable_by_mask(self, layer: nn.Module) -> bool:
        return len(self._zero_channels(layer)) > 0

    def prune_by_mask(self, layer: nn.Module) -> Iterable[int]:
        self._before_mask_pruning(layer)
        layer.zero_grad()

        with torch.no_grad():
            non_zero = self._non_zero_channels(layer)

            w_slices = tuple(
                [tuple(non_zero) if i == 0 else slice(None) for i in range(layer.weight.ndim)]
            )
            b_slices = tuple(
                [tuple(non_zero) if i == 0 else slice(None) for i in range(layer.bias.ndim)]
            )

            self._prune_parameter(layer, "weight", w_slices)
            self._prune_parameter(layer, "bias", b_slices)

        self._after_mask_pruning(layer, non_zero)
        return non_zero

    def prune_by_channels(self, layer: nn.Module, channels: Iterable[int]) -> bool:
        self._befor_channel_pruning(layer, channels)
        layer.zero_grad()

        with torch.no_grad():
            slices = tuple(
                [tuple(channels) if i == 1 else slice(None) for i in range(layer.weight.ndim)]
            )

            self._prune_parameter(layer, "weight", slices)

        self._after_channel_pruning(layer, channels)
        return False

    def _before_mask_pruning(self, layer: nn.Module) -> None:
        pass

    def _befor_channel_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        pass

    def _after_mask_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        pass

    def _after_channel_pruning(self, layer: nn.Module, channels: Iterable[int]) -> None:
        pass

    def _zero_channels(self, layer: nn.Module) -> Iterable[int]:
        if not hasattr(layer, "weight_mask"):
            return []

        dims_to_sum = tuple(range(1, len(layer.weight_mask.shape)))
        mask_sum = layer.weight_mask.sum(dim=dims_to_sum)
        zero_idxs = (mask_sum == 0).nonzero().squeeze().tolist()
        return zero_idxs

    def _non_zero_channels(self, layer: nn.Module) -> Iterable[int]:
        if not hasattr(layer, "weight_mask"):
            return list(range(layer.weight.shape[0]))

        dims_to_sum = tuple(range(1, len(layer.weight_mask.shape)))
        mask_sum = layer.weight_mask.sum(dim=dims_to_sum)
        non_zero_idxs = (mask_sum > 0).nonzero().squeeze().tolist()
        return non_zero_idxs
