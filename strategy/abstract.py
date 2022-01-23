from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from torch import nn
import torch

from scoring.abstract import Scoring


class Strategy(ABC):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__()

        self._scoring = scoring

    @abstractmethod
    def get_mask(self, layer: nn.Module, fraction: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[np.ndarray]:
        pass


class Common:
    weight_mask_name = "weight_mask"
    weight_orig_name = "weight_orig"
    weight_name = "weight"

    @classmethod
    def apply_mask(cls, module: nn.Module) -> torch.Tensor:
        mask = getattr(module, cls.weight_mask_name)
        orig = getattr(module, cls.weight_orig_name)
        pruned = mask * orig
        return pruned


class MaskHook:
    def __init__(self) -> None:
        pass

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        setattr(module, Common.weight_name, Common.apply_mask(module))


class Pruning(ABC):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__()

        self._scoring = scoring

    def prune(self, layers: List[nn.Module], fraction: float) -> None:
        masks = self.__get_masks(layers, fraction)
        masks = [self.__combine_masks(layer, mask) for layer, mask in zip(layers, masks)]
        [self.__set_mask(layer, mask) for layer, mask in zip(layers, masks)]

    def remove(self, layers: List[nn.Module]) -> None:
        [self.__remove_mask(layer) for layer in layers]

    @abstractmethod
    def _get_scores(self, layers: List[nn.Module]) -> List[float]:
        pass

    @abstractmethod
    def _set_mask_value(self, masks: List[np.ndarray], list_idx: int, array_idx: int) -> None:
        pass

    def __get_masks(self, layers: List[nn.Module], fraction: float) -> List[torch.Tensor]:
        scores = self._get_scores(layers)
        list_div_idxs = []

        # Get indexes in flattened and concatenated array that divide individual
        # arrays within the list
        for i in range(len(scores)):
            list_div_idxs.append(sum([scores[j].size for j in range(0, i)]))

        concatenated = np.concatenate(scores)
        sorted_idxs = np.argsort(concatenated, axis=None)

        pruned_fractions = [self.__get_pruned_fraction(layer) for layer in layers]
        p_actual = sum(pruned_fractions) / len(pruned_fractions)
        p = int((sorted_idxs.size - sorted_idxs.size * p_actual) * fraction)
        masks = [np.full(score.shape, True) for score in scores]

        for idx in sorted_idxs[0:p]:
            array_div_idx = max([div_idx for div_idx in list_div_idxs if div_idx <= idx])
            list_idx = list_div_idxs.index(array_div_idx)
            array_idx = idx - array_div_idx
            self._set_mask_value(masks, list_idx, array_idx)

        return [torch.Tensor(mask, layer.weight.device) for mask, layer in zip(masks, layers)]

    def __get_pruned_fraction(self, layer: nn.Module) -> float:
        mask = getattr(layer, Common.weight_name, None)
        fraction = mask[mask == 0].size().numel() / mask.size().numel() if mask is not None else 0.0
        return fraction

    def __combine_masks(self, layer: nn.Module, mask: torch.Tensor) -> torch.Tensor:
        actual = getattr(layer, Common.weight_name, None)
        new_mask = actual * mask if actual is not None else mask
        return new_mask

    def __set_mask(self, layer: nn.Module, mask: torch.Tensor) -> None:
        # Remove old and add new mask into layer
        layer._buffers.pop(Common.weight_mask_name, None)
        layer.register_buffer(Common.weight_mask_name, mask)

        # Register weight_orig into layer's parameters if not registered, yet
        if Common.weight_orig_name not in layer._parameters:
            orig = layer.get_parameter(Common.weight_name)
            layer.register_parameter(Common.weight_orig_name, orig)
            del layer._parameters[Common.weight_name]

        # As we removed weight from layer's parameters, we need to set it
        # as attribute manually
        setattr(layer, Common.weight_name, Common.apply_mask(layer))

        # Register forward hook if not registered, yet
        if not any([isinstance(hook, MaskHook) for hook in layer._forward_pre_hooks.values()]):
            layer.register_forward_pre_hook(MaskHook())

    def __remove_mask(self, layer: nn.Module) -> None:
        weight = Common.apply_mask(layer)

        # Remove mask and orig from layer
        del layer._buffers[Common.weight_mask_name]
        del layer._parameters[Common.weight_orig_name]

        # Remove forward hook
        del_key = next(k for k, hook in layer._forward_pre_hooks if isinstance(hook, MaskHook))
        del layer._forward_pre_hooks[del_key]

        # Set layer's weight
        layer.register_parameter(Common.weight_name, weight)


class StructuredPruning(Pruning):
    def __init__(self) -> None:
        super().__init__()

    def _get_scores(self, layers: List[nn.Module]) -> List[float]:
        scores = [self._scoring.get_score(layer) for layer in layers]
        scores_sum = []

        # Sum along all but first axis to get aggregated score for
        # individual filters/neurons
        for score in scores:
            axes_to_sum = tuple(range(1, len(score.shape)))
            score_sum = np.sum(score, axis=axes_to_sum)
            # Divide score sum by its size to make it comparable across
            # whole model in case of global pruning
            score_sum /= np.array([score[i].size for i in range(score.shape[0])])
            scores_sum.append(score_sum)

        return scores_sum

    def _set_mask_value(self, masks: List[np.ndarray], list_idx: int, array_idx: int) -> None:
        masks[list_idx][array_idx] = False


class UnstructuredPruning(Pruning):
    def __init__(self) -> None:
        super().__init__()

    def _get_scores(self, layers: List[nn.Module]) -> List[float]:
        return [self._scoring.get_score(layer) for layer in layers]

    def _set_mask_value(self, masks: List[np.ndarray], list_idx: int, array_idx: int) -> None:
        np.put(masks[list_idx], array_idx, False)
