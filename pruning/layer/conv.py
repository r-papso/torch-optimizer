from pruning.layer.abstract import StandardPruner


class ConvPruner(StandardPruner):
    def __init__(self) -> None:
        super().__init__()

    def _channel_dim(self) -> int:
        return 1

    def _in_attr_name(self) -> str:
        return "in_channels"

    def _out_attr_name(self) -> str:
        return "out_channels"
