import numpy as np

from abc import abstractmethod
from typing import List
from torch import nn

from structs.layer_mask import LayerMask
from pipeline.pipeline_context import PipelineContext
from pruning.abstract import Pruner
from strategy.abstract import Strategy


class GlobalPruner(Pruner):
    def __init__(
        self, context: PipelineContext, strategy: Strategy, n_steps: int, remove_channels: bool,
    ) -> None:
        super().__init__(context, strategy, n_steps, remove_channels)

    def get_mask(self, layer: nn.Module) -> np.ndarray:
        # TODO: inspiration from here, then delete
        scores = self.get_score(layer)

        # Sum scores along all but first axis
        axes_to_sum = tuple(range(1, len(scores.shape)))
        channel_sum = np.sum(scores, axis=axes_to_sum)
        # Sort channels' indexes by score sum ascending
        sorted_channels = np.argsort(channel_sum)

        # Get pruning fraction
        fraction = self.get_fraction(layer)
        m = int(sorted_channels.size * fraction)
        mask = np.full(scores.shape, True)

        # Get indexes of channels to be pruned and set corresponding values in mask to False
        false_idxs = sorted_channels[0:m]
        mask[false_idxs] = False

        return mask
