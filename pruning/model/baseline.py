from pipeline.pipeline_context import PipelineContext
from pruning.local import LocalPruner
from strategy.abstract import Strategy


class BaselinePruner(LocalPruner):
    def __init__(
        self,
        context: PipelineContext,
        strategy: Strategy,
        n_steps: int,
        fraction: float,
        remove_channels: bool,
    ) -> None:
        super().__init__(context, strategy, n_steps, fraction, remove_channels)
