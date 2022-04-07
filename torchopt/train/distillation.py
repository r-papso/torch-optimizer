from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .loader import DataLoaderWrapper


class KDLoss(nn.Module):
    """Class representing Knowledge Distillation (KD) loss implementation.

    Vanilla KD is based on the architecture of teacher-student model, where already trained 
    big teacher model is used to train smaller student model. During training, student model
    tries to fit probability distribution of teacher's softmax outputs. To measure distance
    between the two probability distributions, Kullback-Leibler (KL) divergence is used. For
    more information, see: https://arxiv.org/pdf/1503.02531.pdf.
    """

    def __init__(
        self,
        teacher: nn.Module,
        train_loader: DataLoaderWrapper,
        test_loader: DataLoaderWrapper,
        device: str,
        T: float,
    ) -> None:
        """Ctor.

        Args:
            teacher (nn.Module): Trained teacher model.
            train_loader (DataLoaderWrapper): Training set.
            test_loader (DataLoaderWrapper): Testing set.
            device (str): Name of the device where training/testing will be performed.
            T (float): Temperature.
        """
        super().__init__()

        self._train = train_loader
        self._test = test_loader
        self._teacher = self._init_teacher(teacher)
        self._device = device
        self._kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._T = T

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        loader = self._train if self._train.timestamp() > self._test.timestamp() else self._test
        data, labels = loader.cahced_batch()
        data, labels = data.to(self._device), labels.to(self._device)

        assert torch.all(labels == targets).item()

        preds = self._teacher(data)
        preds_log = F.log_softmax(preds / self._T, dim=1)
        inputs_log = F.log_softmax(inputs / self._T, dim=1)

        kl_loss = self._kl_loss(inputs_log, preds_log)
        ce_loss = F.cross_entropy(inputs, targets)

        return kl_loss + ce_loss

    def _init_teacher(self, teacher: nn.Module) -> nn.Module:
        teacher_cpy = deepcopy(teacher)

        for param in teacher_cpy.parameters():
            param.requires_grad = False

        return teacher_cpy.eval()
