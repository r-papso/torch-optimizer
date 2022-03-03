from abc import ABC, abstractmethod
from typing import Iterable
from .context import PipelineContext


class PipelineStep(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        pass


class PipelineContainer(PipelineStep):
    def __init__(self, steps: Iterable[PipelineStep]) -> None:
        super().__init__()
        self.__steps = steps

    def run(self, context: PipelineContext) -> None:
        for step in self.__steps:
            step.run(context)


class TrainStep(PipelineStep):
    def __init__(self, epochs: int) -> None:
        super().__init__()

    def run(self, context: PipelineContext) -> None:
        pass


class TestStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__()

    def run(self, context: PipelineContext) -> None:
        pass


class PruneStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__()

    def run(self, context: PipelineContext) -> None:
        return super().run(context)
