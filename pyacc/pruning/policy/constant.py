from torch import nn
from pyacc.pruning.policy.abstract import Policy


class ConstantPolicy(Policy):
    def __init__(self, fraction: float) -> None:
        super().__init__()

        self.__fraction = fraction

    def get_fraction(self, layer: nn.Module) -> float:
        return self.__fraction
