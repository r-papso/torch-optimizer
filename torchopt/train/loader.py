import time

from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader


class DataLoaderWrapper:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._cahced = None
        self._time = -1.0

    def __iter__(self):
        for batch in self._loader:
            self._cahced = batch
            self._time = time.time()
            yield batch

        self._cahced = None

    def __len__(self):
        return len(self._loader)

    def cahced_batch(self) -> Tuple[Tensor, Tensor]:
        return self._cahced

    def timestamp(self) -> float:
        return self._time
