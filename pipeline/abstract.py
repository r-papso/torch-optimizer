from abc import ABC

from pipeline.pipeline_context import PipelineContext


class PipelineEntity(ABC):
    def __init__(self, context: PipelineContext) -> None:
        super().__init__()

        self._context = context
