from abc import ABC, abstractmethod
from typing import List

from torch import nn
import torch


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prune_by_mask(self, layer: nn.Module) -> List[int]:
        pass

    @abstractmethod
    def prune_by_channels(self, layer: nn.Module, channels: List[int]) -> bool:
        pass

    def _prune_parameter(self, layer: nn.Module, name: str, mask: torch.Tensor) -> None:
        if (param := layer._parameters.get(name)) is not None:
            param.data = param.data[mask]

        if (param_mask := layer._buffers.get(f"{name}_mask")) is not None:
            layer._buffers[f"{name}_mask"] = param_mask[mask]

        if (param_orig := layer._parameters.get(f"{name}_orig")) is not None:
            param_orig.data = param_orig.data[mask]
