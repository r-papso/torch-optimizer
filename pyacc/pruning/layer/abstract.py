from abc import ABC, abstractmethod
from typing import Iterable

from pyacc.pruning.prune import apply_mask
from torch import nn


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prunable_by_mask(self, layer: nn.Module) -> bool:
        pass

    @abstractmethod
    def prune_by_mask(self, layer: nn.Module) -> Iterable[int]:
        pass

    @abstractmethod
    def prune_by_channels(self, layer: nn.Module, channels: Iterable[int]) -> bool:
        pass

    def _prune_parameter(self, layer: nn.Module, name: str, slices: Iterable[slice]) -> None:
        if (param := layer._parameters.get(name)) is not None:
            param.data = param.data[slices]

        if (param_mask := layer._buffers.get(f"{name}_mask")) is not None:
            layer._buffers[f"{name}_mask"] = param_mask[slices]

        if (param_orig := layer._parameters.get(f"{name}_orig")) is not None:
            param_orig.data = param_orig.data[slices]

        if param_mask is not None and param_orig is not None:
            setattr(layer, name, apply_mask(layer, name))
