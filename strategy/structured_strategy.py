import numpy as np

from typing import List
from torch import nn

from dataobject.layer_mask import LayerMask
from pipeline.pipeline_context import PipelineContext
from strategy.abstract import Strategy


class StructuredStrategy(Strategy):
    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context)

    def get_masks(self, fraction: float, layers: List[nn.Module]) -> List[LayerMask]:
        # TODO: implement
        pass
