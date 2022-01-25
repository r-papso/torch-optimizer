from abc import abstractmethod
from typing import List
import numpy as np
from torch import nn
import torch
from pruning.layer.abstract import LayerPruner


class StandardPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module) -> List[int]:
        assert isinstance(layer, nn.Linear), f"Invalid layer type: {type(layer)}."

        layer.zero_grad()
        with torch.no_grad():
            zero_channels = self.__get_zero_channels(layer)

            w_mask = torch.ones_like(layer.weight, dtype=torch.bool)
            w_mask[zero_channels] = False
            b_mask = torch.ones_like(layer.bias, dtype=torch.bool)
            b_mask[zero_channels] = False

            self._prune_parameter(layer, "weight", w_mask)
            self._prune_parameter(layer, "bias", b_mask)

        out_channels = getattr(layer, self.__out_attr_name) - len(zero_channels)
        setattr(layer, self.__out_attr_name, out_channels)
        return zero_channels

    def prune_by_channels(self, layer: nn.Module, channels: List[int]) -> bool:
        assert isinstance(layer, nn.Linear), f"Invalid layer type: {type(layer)}."

        layer.zero_grad()
        with torch.no_grad():
            w_mask = torch.ones_like(layer.weight, dtype=torch.bool)

            slices = [slice(None)] * w_mask.ndim
            slices[self.__channel_dim] = np.array(channels)
            slices = tuple(slices)

            w_mask[slices] = False
            self._prune_parameter(layer, "weight", w_mask)

        in_channels = getattr(layer, self.__in_attr_name) - len(channels)
        setattr(layer, self.__in_attr_name, in_channels)
        return False

    @abstractmethod
    def _channel_dim(self) -> int:
        pass

    @abstractmethod
    def _in_attr_name(self) -> str:
        pass

    @abstractmethod
    def _out_attr_name(self) -> str:
        pass

    def __get_zero_channels(self, layer: nn.Module) -> List[int]:
        dims_to_sum = tuple(range(1, len(layer.weight_mask.shape)))
        mask_sum = layer.weight_mask.sum(dim=dims_to_sum)
        zero_idxs = (mask_sum == 0).nonzero().squeeze().tolist()
        return zero_idxs
