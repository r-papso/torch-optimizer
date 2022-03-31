import os
import shutil
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from ignite.handlers.param_scheduler import LRScheduler
from thop import profile
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from . import utils
from .optim.constraint import ChannelConstraint
from .optim.objective import (
    Accuracy,
    AccuracyFinetuned,
    Macs,
    MacsPenalty,
    Objective,
    ObjectiveContainer,
)
from .optim.optimizer import BinaryGAOptimizer, IntegerGAOptimizer, Optimizer
from .prune.pruner import ChannelPruner, Pruner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (1, 3, 32, 32)


def vgg_best(finetune: bool, mode: str, dropout_decay: float = 0.0, **kwargs) -> nn.Module:
    if mode not in ["int", "binary"]:
        raise ValueError("Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    model = utils.get_vgg16()
    conv_names = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    pruner = ChannelPruner(conv_names, INPUT_SHAPE)

    optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
    objective = _objective_best(model, pruner, finetune, kwargs.get("weight", 1.0))
    constraint = ChannelConstraint(model=model, pruner=pruner)
    solution = optim.maximize(objective, constraint)

    model = pruner.prune(model, solution)
    model = _reduce_dropout(model, dropout_decay)
    model = _train(model)

    return model


def vgg_constrained(
    finetune: bool,
    mode: str,
    bounds: Iterable,
    output_dir: str,
    dropout_decay: float = 0.0,
    **kwargs,
) -> nn.Module:
    if mode not in ["int", "binary"]:
        raise ValueError("Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    model = utils.get_vgg16()
    conv_names = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    pruner = ChannelPruner(conv_names, INPUT_SHAPE)

    # Iteratively prune model according to upper bounds
    for b in bounds:
        optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
        objective = _objective_constrained(model, pruner, finetune, b, kwargs.get("weight", 1.0))
        constraint = ChannelConstraint(model=model, pruner=pruner)
        solution = optim.maximize(objective, constraint)

        model = pruner.prune(model, solution)
        model = _reduce_dropout(model, dropout_decay)
        model = _train(model)
        torch.save(model, os.path.join(output_dir, f"vgg_constrained_{b}.pth"))

    return model


def resnet_best(finetune: bool, mode: str, **kwargs) -> nn.Module:
    pass


def resnet_constrained(
    finetune: bool, mode: str, bounds: Iterable, output_dir: str, **kwargs
) -> nn.Module:
    pass


def _optimization_data() -> Tuple[Iterable, Iterable, Iterable]:
    train_loader, val_loader, test_loader = utils.cifar10_loaders(
        folder="./data/cifar10",
        batch_size=256,
        val_size=5000,
        train_transform=Compose([ToTensor()]),
        test_transform=Compose([ToTensor()]),
    )

    train_data = utils.loader_to_memory(train_loader, DEVICE)
    val_data = utils.loader_to_memory(val_loader, DEVICE)
    test_data = utils.loader_to_memory(test_loader, DEVICE)

    return train_data, val_data, test_data


def _train_data() -> Tuple[Iterable, Iterable, Iterable]:
    train_loader, val_loader, test_loader = utils.cifar10_loaders(
        folder="./data/cifar10",
        batch_size=256,
        val_size=5000,
        train_transform=Compose([RandomHorizontalFlip(p=0.5), RandomCrop(32, 4), ToTensor()]),
        test_transform=Compose([ToTensor()]),
    )

    return train_loader, val_loader, test_loader


def _objective_best(model: nn.Module, pruner: Pruner, finetune: bool, w: float) -> Objective:
    train_data, val_data, test_data = _optimization_data()

    orig_acc = utils.evaluate(model, test_data, DEVICE)
    orig_macs, _ = profile(model, inputs=(torch.randn(INPUT_SHAPE, device=DEVICE),), verbose=False)

    acc = (
        Accuracy(model, pruner, 1.0, val_data, orig_acc)
        if not finetune
        else AccuracyFinetuned(model, pruner, 1.0, train_data, val_data, len(train_data), orig_acc)
    )
    macs = Macs(model, pruner, orig_macs, w, in_shape=INPUT_SHAPE)

    return ObjectiveContainer(acc, macs)


def _objective_constrained(
    model: nn.Module, pruner: Pruner, finetune: bool, p: float, w: float
) -> Objective:
    train_data, val_data, test_data = _optimization_data()

    orig_acc = utils.evaluate(model, test_data, DEVICE)
    orig_macs, _ = profile(model, inputs=(torch.randn(INPUT_SHAPE, device=DEVICE),), verbose=False)

    acc = (
        Accuracy(model, pruner, 1.0, val_data, orig_acc)
        if not finetune
        else AccuracyFinetuned(model, pruner, 1.0, train_data, val_data, len(train_data), orig_acc)
    )
    macs = MacsPenalty(model, pruner, w, p, orig_macs, INPUT_SHAPE)

    return ObjectiveContainer(acc, macs)


def _integer_GA(model: nn.Module, **kwargs) -> Optimizer:
    bounds = [
        (0, len(module.weight) - 1) for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    pop_size = kwargs.get("pop_size", 100)
    ind_size = len(bounds)

    return IntegerGAOptimizer(
        ind_size=ind_size,
        pop_size=pop_size,
        elite_num=kwargs.get("elite_num", int(0.1 * pop_size)),
        tourn_size=kwargs.get("tourn_size", int(0.1 * pop_size)),
        n_gen=kwargs.get("n_gen", 50),
        mutp=kwargs.get("mutp", 0.1),
        mut_indp=kwargs.get("mut_indp", 0.05),
        cx_indp=kwargs.get("cx_indp", 0.5),
        bounds=bounds,
    )


def _binary_GA(model: nn.Module, **kwargs):
    pop_size = kwargs.get("pop_size", 100)
    ind_size = sum(
        [len(module.weight) for module in model.modules() if isinstance(module, nn.Conv2d)]
    )

    return BinaryGAOptimizer(
        ind_size=ind_size,
        pop_size=pop_size,
        elite_num=kwargs.get("elite_num", int(0.1 * pop_size)),
        tourn_size=kwargs.get("tourn_size", int(0.1 * pop_size)),
        n_gen=kwargs.get("n_gen", 50),
        mutp=kwargs.get("mutp", 0.1),
        mut_indp=kwargs.get("mut_indp", 0.01),
        cx_indp=kwargs.get("cx_indp", 0.5),
    )


def _train(model: nn.Module) -> nn.Module:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(package_dir, "checkpoint")

    if os.path.exists(checkpoint):
        shutil.rmtree(checkpoint)

    train_set, val_set, test_set = _train_data()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    torch_lr_scheduler = CosineAnnealingLR(optimizer, 50)
    scheduler = LRScheduler(torch_lr_scheduler)

    _ = utils.train_ignite(
        model=model,
        train_set=train_set,
        test_set=test_set,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=50,
        checkpoint_path=checkpoint,
        lr_scheduler=scheduler,
    )

    model_f = next(f for f in os.listdir(checkpoint) if os.path.isfile(os.path.join(checkpoint, f)))
    model.load_state_dict(torch.load(model_f))

    return model


def _reduce_dropout(model: nn.Module, do_decay: float) -> nn.Module:
    for module in [module for module in model.modules() if isinstance(module, nn.Dropout)]:
        module.p -= do_decay
