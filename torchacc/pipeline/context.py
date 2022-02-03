from torch import nn
from torch.utils.data import DataLoader


class PipelineContext:
    def __init__(self, model: nn.Module, train_data: DataLoader, test_data: DataLoader) -> None:
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
