from typing import List
import numpy as np
from torch import nn

from structs.layer_mask import LayerMask
from strategy.abstract import Strategy


class UnstructuredStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def get_mask(self, layer: nn.Module, fraction: float) -> LayerMask:
        score = self._scoring.get_score(layer)
        sorted_idxs = np.argsort(score, axis=None)

        n_zeros = int(sorted_idxs.size * fraction)
        mask = np.full(score.shape, True)
        np.put(mask, sorted_idxs[0:n_zeros], False)

        # TODO: return mask or implement layer masking here

        return mask

    def get_masks(self, layers: List[nn.Module], fraction: float) -> List[LayerMask]:
        scores = self._scoring.get_scores(layers)

        # Get indexes in flattened and concatenated array that divide individual
        # arrays within the list
        array_div_idxs = []
        for i in range(len(scores)):
            array_div_idxs.append(sum([scores[j].size for j in range(0, i)]))

        concatenated = np.concatenate([score.flatten() for score in scores])
        sorted_idxs = np.argsort(concatenated, axis=None)

        n_zeros = int(sorted_idxs.size * fraction)
        masks = [np.full(score.shape, True) for score in scores]

        for idx in sorted_idxs[0:n_zeros]:
            array_div_idx = max([div_idx for div_idx in array_div_idxs if div_idx <= idx])
            list_idx = array_div_idxs.index(array_div_idx)
            array_idx = idx - array_div_idx
            np.put(masks[list_idx], array_idx, False)

        return masks
