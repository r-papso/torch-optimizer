from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import nn
import torch


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module) -> List[Tuple[int, ...]]:
        assert self._prunable_by_mask(), f"{LayerPruner} -> Pruning by mask is not supported."

        layer.zero_grad()

        with torch.no_grad():
            for name, mask in self._params_masks(layer):
                self.__prune_parameter(layer, name, mask)

            self._additional_mask_pruning(layer)
            return self._get_output_shape(layer)

    def prune_by_input(self, layer: nn.Module, input: List[Tuple[int, ...]]) -> None:
        assert self._prunable_by_input(), f"{LayerPruner} -> Pruning by input is not supported."

        layer.zero_grad()

        with torch.no_grad():
            for name, mask in self._params_masks_by_input(input):
                self.__prune_parameter(layer, name, mask)

            self._additional_input_pruning(layer, input)

    @abstractmethod
    def _params_masks(self, layer: nn.Module) -> List[Tuple[str, torch.Tensor]]:
        pass

    @abstractmethod
    def _params_masks_by_input(
        self, layer: nn.Module, input: List[Tuple[int, ...]]
    ) -> List[Tuple[str, torch.Tensor]]:
        pass

    @abstractmethod
    def _get_output_shape(self) -> List[Tuple[int, ...]]:
        pass

    @abstractmethod
    def _prunable_by_mask(self) -> bool:
        pass

    @abstractmethod
    def _prunable_by_input(self, layer: nn.Module) -> bool:
        pass

    def _additional_mask_pruning(self, layer: nn.Module) -> None:
        pass

    def _additional_input_pruning(self, layer: nn.Module, input: List[Tuple[int, ...]]) -> None:
        pass

    def __prune_parameter(
        self, layer: nn.Module, param_name: str, param_mask: torch.Tensor
    ) -> None:
        if (param := layer._parameters.get(param_name)) is not None:
            param.data = param.data[param_mask]

        if (mask := layer._buffers.get(f"{param_name}_mask")) is not None:
            layer._buffers[f"{param_name}_mask"] = mask[param_mask]

        if (param_orig := layer._parameters.get(f"{param_name}_orig")) is not None:
            param_orig.data = param_orig.data[param_mask]


class StandardLayerPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def _params_masks(self, layer: nn.Module) -> List[Tuple[str, torch.Tensor]]:
        # TODO implement
        pass

    def _params_masks_by_input(
        self, layer: nn.Module, input: List[Tuple[int, ...]]
    ) -> List[Tuple[str, torch.Tensor]]:
        # TODO implement
        pass

    def _prunable_by_mask(self) -> bool:
        return True

    def _prunable_by_input(self, layer: nn.Module) -> bool:
        return True


class LayerPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module) -> List[Tuple[int, ...]]:
        assert (
            self._can_prune_by_mask()
        ), f"{LayerPruner} -> Pruner implementation does not support pruning by mask."
        assert hasattr(layer, "weight_mask"), f"{LayerPruner} -> Weight mask in layer not set."

        layer.zero_grad()
        with torch.no_grad():
            zero_neurons = self.__get_zero_neurons(layer)

            w_mask = torch.ones_like(layer.weight, dtype=torch.bool)
            w_mask[zero_neurons] = False
            b_mask = torch.ones_like(layer.bias, dtype=torch.bool)
            b_mask[zero_neurons] = False

            self.__prune_parameter(layer, "weight", w_mask)
            self.__prune_parameter(layer, "bias", b_mask)
            self._additional_mask_pruning(layer)

            return zero_neurons

    @abstractmethod
    def prune_by_indicies(self, layer: nn.Module, indicies: List[int]) -> None:
        pass

    def __get_zero_neurons(self, layer: nn.Module) -> List[int]:
        axes_to_sum = tuple(range(1, len(layer.weight_mask.shape)))
        neurons_sum = layer.weight_mask.sum(axes_to_sum)
        zero_indicies = (neurons_sum == 0).nonzero().squeeze().tolist()
        return zero_indicies

    def __prune_parameter(
        self, layer: nn.Module, param_name: str, param_mask: torch.Tensor
    ) -> None:
        if (param := layer._parameters.get(param_name)) is not None:
            param.data = param.data[param_mask]

        if (mask := layer._buffers.get(f"{param_name}_mask")) is not None:
            layer._buffers[f"{param_name}_mask"] = mask[param_mask]

        if (param_orig := layer._parameters.get(f"{param_name}_orig")) is not None:
            param_orig.data = param_orig.data[param_mask]

    @abstractmethod
    def _can_prune_by_mask(self) -> bool:
        pass

    @abstractmethod
    def _mask_pruning_weight_mask(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def _mask_pruning_bias_mask(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def _indicies_pruning_weight_mask(self, layer: nn.Module, indicies: List[int]) -> torch.Tensor:
        pass

    @abstractmethod
    def _indicies_pruning_bias_mask(self, layer: nn.Module, indicies: List[int]) -> torch.Tensor:
        pass

    def _additional_mask_pruning(self, layer: nn.Module) -> None:
        pass

    def _additional_indicies_pruning(self, layer: nn.Module, indicies: List[int]) -> None:
        pass
