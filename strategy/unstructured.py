from typing import List
import numpy as np
from torch import nn

from scoring.abstract import Scoring
from strategy.abstract import Strategy


class UnstructuredStrategy(Strategy):
    def __init__(self, scoring: Scoring) -> None:
        super().__init__(scoring)

    def get_mask(self, layer: nn.Module, fraction: float) -> np.ndarray:
        score = self._scoring.get_score(layer)
        sorted_idxs = np.argsort(score, axis=None)

        p = int(sorted_idxs.size * fraction)
        mask = np.full(score.shape, True)
        np.put(mask, sorted_idxs[0:p], False)

        return mask

    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[np.ndarray]:
        scores = [self._scoring.get_score(layer) for layer in layers]

        # Get indexes in flattened and concatenated array that divide individual
        # arrays within the list
        array_div_idxs = []
        for i in range(len(scores)):
            array_div_idxs.append(sum([scores[j].size for j in range(0, i)]))

        concatenated = np.concatenate([score.flatten() for score in scores])
        sorted_idxs = np.argsort(concatenated, axis=None)

        p = int(sorted_idxs.size * fraction)
        masks = [np.full(score.shape, True) for score in scores]

        for idx in sorted_idxs[0:p]:
            array_div_idx = max([div_idx for div_idx in array_div_idxs if div_idx <= idx])
            list_idx = array_div_idxs.index(array_div_idx)
            array_idx = idx - array_div_idx
            np.put(masks[list_idx], array_idx, False)

        return masks
