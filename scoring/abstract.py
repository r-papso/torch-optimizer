import numpy as np

from typing import List
from abc import abstractmethod
from torch import nn

from pipeline.abstract import PipelineEntity
from pipeline.pipeline_context import PipelineContext


class Scorer(PipelineEntity):
    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context)

    @abstractmethod
    def get_score(self, layers: List[nn.Module]) -> List[np.ndarray]:
        pass
