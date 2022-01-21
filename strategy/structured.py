from typing import List
import numpy as np
from torch import nn

from scoring.abstract import Scoring
from strategy.abstract import Strategy


class StructuredStrategy(Strategy):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__(scoring)

    def get_mask(self, layer: nn.Module, fraction: float) -> np.ndarray:
        score = self._scoring.get_score(layer)

        # Sum along all but first axis to get aggregated score for
        # individual filters/neurons
        axes_to_sum = tuple(range(1, len(score.shape)))
        score_sum = np.sum(score, axis=axes_to_sum)
        sorted_idxs = np.argsort(score_sum, axis=None)

        p = int(sorted_idxs.size * fraction)
        mask = np.full(score.shape, True)
        false_idxs = sorted_idxs[0:p]
        mask[false_idxs] = False

        return mask

    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[np.ndarray]:
        scores = [self._scoring.get_score(layer) for layer in layers]
        scores_sum = []
        array_div_idxs = []

        # Sum along all but first axis to get aggregated score for
        # individual filters/neurons
        for score in scores:
            axes_to_sum = tuple(range(1, len(score.shape)))
            score_sum = np.sum(score, axis=axes_to_sum)
            scores_sum.append(score_sum)

        # Get indexes within the score sum arrays that divide individual
        # arrays within the list
        for i in range(len(scores_sum)):
            array_div_idxs.append(sum([scores_sum[j].size for j in range(0, i)]))

        concatenated = np.concatenate(scores_sum)
        sorted_idxs = np.argsort(concatenated, axis=None)

        p = int(sorted_idxs.size * fraction)
        masks = [np.full(score.shape, True) for score in scores]

        for idx in sorted_idxs[0:p]:
            array_div_idx = max([div_idx for div_idx in array_div_idxs if div_idx <= idx])
            list_idx = array_div_idxs.index(array_div_idx)
            array_idx = idx - array_div_idx
            masks[list_idx][array_idx] = False

        return masks
