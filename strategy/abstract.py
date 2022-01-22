from abc import ABC, abstractmethod
from typing import List
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


class Pruning(ABC):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__()

        self._scoring = scoring

    def apply_masks(self, layers: List[nn.Module], fraction: float) -> None:
        masks = self.__get_masks(layers, fraction)
        [self.__combine_masks(layer, mask) for layer, mask in zip(layers, masks)]

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

    def __get_actual_mask(self, layer: nn.Module) -> torch.Tensor:
        try:
            return layer.get_buffer("mask")
        except Exception:
            return None

    def __get_pruned_fraction(self, layer: nn.Module) -> float:
        mask = self.__get_actual_mask(layer)
        return mask[mask == 0].size().numel() / mask.size().numel() if mask is not None else 0.0

    def __combine_masks(self, layer: nn.Module, mask: torch.Tensor) -> None:
        actual = self.__get_actual_mask(layer)

        if actual:
            new_mask = actual * mask
            del layer._buffers["mask"]
        else:
            new_mask = mask

        layer.register_buffer("mask", new_mask)


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
