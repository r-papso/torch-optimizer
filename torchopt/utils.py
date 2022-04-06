import os
import sys
from datetime import datetime
from typing import Callable, Iterable, Tuple

import ignite.metrics as metrics
import torch
import torch.nn as nn
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10

from .train.loader import DataLoaderWrapper

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PACKAGE_DIR, "model"))


def get_vgg16() -> nn.Module:
    """Returns trained VGG16 model on CIFAR10 dataset with 92.25% accuracy. Model was 
    trained on subset of CIFAR10 training set consisting of first 45 000 samples.

    Returns:
        nn.Module: Trained VGG16 model on CIFAR10 dataset.
    """
    model = torch.load(os.path.join(PACKAGE_DIR, "model", "vgg16_cifar10_0.9225_45k.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def get_resnet56() -> nn.Module:
    """Returns trained ResNet56 model on CIFAR10 dataset with 93.20% accuracy. Model was 
    trained on subset of CIFAR10 training set consisting of first 45 000 samples.

    Returns:
        nn.Module: Trained ResNet56 model on CIFAR10 dataset.
    """
    model = torch.load(os.path.join(PACKAGE_DIR, "model", "resnet56_cifar10_0.9320_45k.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def cifar10_loaders(
    folder: str,
    batch_size: int,
    val_size: int,
    train_transform: Callable,
    test_transform: Callable,
) -> Tuple[DataLoaderWrapper, ...]:
    """Returns training, validation and testing data loader for CIFAR10 dataset. Trainig data 
    loader contains (50 000 - val_size) samples, validation data loader contains val_size samples 
    and testing data loader contains 10 000 samples.

    Args:
        folder (str): Path to the folder where CIFAR10 dataset will be downloaded.
        batch_size (int): Batch size.
        val_size (int): Validation set size.
        train_transform (Callable): Transform applied to data contained in training set.
        test_transform (Callable): Transform applied to data contained in testing set.

    Returns:
        Tuple[DataLoaderWrapper, ...]: Training, validation and testing data loader.
    """
    train_set = CIFAR10(download=True, root=folder, transform=train_transform, train=True)
    test_set = CIFAR10(download=False, root=folder, transform=test_transform, train=False)

    idxs = list(range(len(train_set)))
    split = len(train_set) - val_size
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return (
        DataLoaderWrapper(train_loader),
        DataLoaderWrapper(val_loader),
        DataLoaderWrapper(test_loader),
    )


def loader_to_memory(data_loader: Iterable, device: str) -> Iterable[Tuple[Tensor, Tensor]]:
    """Copies data contained in data loader to memory of the device specified as a parameter.

    Args:
        data_loader (Iterable): Data loader data of which will be copied to the specified memory.
        device (str): Name of the device, can be one of ('cuda', 'cpu').

    Returns:
        Iterable[Tuple[Tensor, Tensor]]: List of minibatches of samples copied to the specified memory.
    """
    return [(inputs.to(device), labels.to(device)) for inputs, labels in data_loader]


def train_ignite(
    model: nn.Module,
    train_set: Iterable[Tuple[Tensor, Tensor]],
    test_set: Iterable[Tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: Callable,
    epochs: int,
    checkpoint_path: str = None,
    lr_scheduler: LRScheduler = None,
) -> dict:
    """Trains model using torch-ignite framework.

    Args:
        model (nn.Module): Model to be trained.
        train_set (Iterable[Tuple[Tensor, Tensor]]): Training set.
        test_set (Iterable[Tuple[Tensor, Tensor]]): Testing set.
        optimizer (Optimizer): Optimizer used in training.
        loss_fn (Callable): Loss function to be minimized.
        epochs (int): Number of training epochs.
        checkpoint_path (str, optional): Checkpoint path. Defaults to None.
        lr_scheduler (LRScheduler, optional): Learning rate scheduler. Defaults to None.

    Returns:
        dict: Training history.
    """
    metric_dict = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device, output_transform=_custom_output_transform
    )

    # val_metrics = {"accuracy": metrics.Accuracy(), "loss": metrics.Loss(loss_fn)}
    val_metrics = {"accuracy": metrics.Accuracy()}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_train, metric_dict, optimizer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_test, evaluator, test_set, metric_dict)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_time)

    if lr_scheduler is not None:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    if checkpoint_path is not None:
        checkpoint = _create_checkpoint(model, trainer, checkpoint_path)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint)

    trainer.run(train_set, max_epochs=epochs)
    return metric_dict


def train(
    model: nn.Module,
    data: Iterable[Tuple[Tensor, Tensor]],
    device: str,
    optimizer: Optimizer,
    loss_fn: Callable,
    iterations: int,
) -> nn.Module:
    """Trains model using standard training loop.

    Args:
        model (nn.Module): Model to be trained.
        data (Iterable[Tuple[Tensor, Tensor]]): Training set.
        device (str): Name of the device where training will be performed.
        optimizer (Optimizer): Optimizer used in training.
        loss_fn (Callable): Loss function to be minimized.
        iterations (int): Number of training iterations (number of total minibatches).

    Returns:
        nn.Module: Trained model.
    """
    model = model.train().to(device)
    iters = 0

    while iters < iterations:
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            iters += 1
            if iters == iterations:
                break

    return model


def evaluate(model: nn.Module, data: Iterable[Tuple[Tensor, Tensor]], device: str) -> float:
    """Evaluates model's accuracy.

    Args:
        model (nn.Module): Model to be evaluated.
        data (Iterable[Tuple[Tensor, Tensor]]): Dataset on which evaluation will be done.
        device (str): Name of the device where evaluation will be performed.

    Returns:
        float: Accuracy of the model on the specified dataset.
    """
    model = model.eval().to(device)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total


def prunable_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Returns all prunable layers of the model.

    This method returns all model's convolutional and linear layers except the model's
    classifier (last layer of the model). 

    Args:
        model (nn.Module): Model from which prunable modules will be returned.

    Returns:
        Iterable[Tuple[str, nn.Module]]: All prunable layers of the model.
    """
    last_layer = list(model.modules())[-1]

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module is not last_layer:
            yield name, module


def count_params(model: nn.Module) -> int:
    """Returns total number of model's trainable parameters.

    Args:
        model (nn.Module): Model in which parameters will be counted.

    Returns:
        int: Total number of model's trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_params(model: nn.Module) -> nn.Module:
    """Randomly reinitializes all model's trainable parameters.

    Modules's reinitialization is achieved by calling its reset_parameters function.

    Args:
        model (nn.Module): Model in which parameters will be reset.

    Returns:
        nn.Module: Model with reinitialized parameters.
    """
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    return model


def _custom_output_transform(x, y, y_pred, loss):
    return {"y": y, "y_pred": y_pred, "loss": loss.item(), "criterion_kwargs": {}}


def _log_train(engine: Engine, metric_dict: dict, optimizer: Optimizer) -> None:
    epoch = engine.state.epoch
    metrics = engine.state.metrics
    acc = metrics["accuracy"]
    # loss = metrics["loss"]
    time = datetime.now().strftime("%H:%M:%S")
    lr = optimizer.param_groups[0]["lr"]

    # metric_dict[epoch] = {"train_acc": acc, "train_loss": loss}
    metric_dict[epoch] = {"train_acc": acc}
    print(
        # f"{time} - Epoch: {epoch:04d} Train accuracy: {acc:.4f} Train loss: {loss:.4f} LR: {lr:.8f}"
        f"{time} - Epoch: {epoch:04d} Train accuracy: {acc:.4f} LR: {lr:.8f}"
    )


def _log_test(engine: Engine, evaluator: Engine, test_set: Iterable, metric_dict: dict) -> None:
    evaluator.run(test_set)

    epoch = engine.state.epoch
    metrics = evaluator.state.metrics
    acc = metrics["accuracy"]
    # loss = metrics["loss"]
    time = datetime.now().strftime("%H:%M:%S")

    # metric_dict[epoch].update({"test_acc": acc, "test_loss": loss})
    metric_dict[epoch].update({"test_acc": acc})
    # print(f"{time} - Epoch: {epoch:04d} Test accuracy:  {acc:.4f} Test loss:  {loss:.4f}")
    print(f"{time} - Epoch: {epoch:04d} Test accuracy:  {acc:.4f}")


def _log_time(engine: Engine) -> None:
    name = engine.last_event_name.name
    time = engine.state.times[name]
    print(f"{name} took {time:.4f} seconds")


def _create_checkpoint(model: nn.Module, trainer: Engine, path: str) -> Checkpoint:
    to_save = {"model": model}
    return Checkpoint(
        to_save,
        path,
        n_saved=1,
        filename_prefix="best",
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
        greater_or_equal=True,
    )
