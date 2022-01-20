from abc import abstractmethod
from typing import List
from torch import nn

from structs.layer_mask import LayerMask
from pipeline.pipeline_context import PipelineContext
from pruning.abstract import Pruner
from strategy.abstract import Strategy


class LocalPruner(Pruner):
    def __init__(
        self, context: PipelineContext, strategy: Strategy, n_steps: int, remove_channels: bool,
    ) -> None:
        super().__init__(context, strategy, n_steps, remove_channels)

    @abstractmethod
    def get_fraction(self, layer: nn.Module) -> float:
        pass

    def get_masks(self, layers: List[nn.Module]) -> List[LayerMask]:
        # TODO: implement
        pass
