from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple


class Constraint(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def feasible(self, solution: Any) -> bool:
        pass


class ConstraintContainer(Constraint):
    def __init__(self, *constraints: Constraint) -> None:
        super().__init__()

        self._constrs = constraints

    def feasible(self, solution: Any) -> bool:
        return all(constr.feasible(solution) for constr in self._constrs)


class ChannelConstraint(Constraint):
    def __init__(self, channel_map: Dict[str, Tuple[int, int]]) -> None:
        super().__init__()

        self._map = channel_map

    def feasible(self, solution: Any) -> bool:
        for start, lenght in self._map.values():
            if not any(solution[start : start + lenght]):
                return False

        return True


class LZeroNorm(Constraint):
    def __init__(self, left_operand: int, comparer: Callable[[int, int], bool]) -> None:
        super().__init__()

        self._left_operand = left_operand
        self._comparer = comparer

    def feasible(self, solution: Any) -> bool:
        return self._comparer(sum(solution), self._left_operand)
