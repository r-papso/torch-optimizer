from abc import abstractmethod

from pipeline.abstract import PipelineEntity
from pipeline.pipeline_context import PipelineContext


class Trainer(PipelineEntity):
    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context)

    @abstractmethod
    def before_pruning_train(self) -> None:
        pass

    @abstractmethod
    def during_pruning_train(self) -> None:
        pass

    @abstractmethod
    def after_pruning_train(self) -> None:
        pass
