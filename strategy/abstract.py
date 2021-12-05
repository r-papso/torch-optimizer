import numpy as np

from abc import abstractmethod
from typing import List
from torch import nn

from dataobject.layer_mask import LayerMask
from pipeline.pipeline_context import PipelineContext
from pipeline.abstract import PipelineEntity


class Strategy(PipelineEntity):
    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context)

    @abstractmethod
    def get_masks(self, fraction: float, layers: List[nn.Module]) -> List[LayerMask]:
        pass

