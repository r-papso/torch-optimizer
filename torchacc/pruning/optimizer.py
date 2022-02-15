from copy import deepcopy
import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import torch.nn as nn
from deap import algorithms, base, creator, tools
from deap.base import Toolbox

from .evaluator import Evaluator
from .pruner import Pruner


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def optimize(self, model: nn.Module, pruner: Pruner, evaluator: Evaluator) -> nn.Module:
        pass


class GAOptimizer(Optimizer):
    def __init__(
        self, individual_length: int, pop_size: int, tournsize: int, n_gen: int, mutpb: float
    ) -> None:
        super().__init__()

        self._indl = individual_length
        self._pop_size = pop_size
        self._tournsize = tournsize
        self._n_gen = n_gen
        self._mutpb = mutpb

    def optimize(self, model: nn.Module, pruner: Pruner, evaluator: Evaluator) -> nn.Module:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()

        tb.register("attr_bool", random.randint, 0, 1)
        tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bool, n=self._indl)
        tb.register("population", tools.initRepeat, list, tb.individual)

        def evaluate(individual) -> Tuple[float]:
            model_cpy = deepcopy(model)
            model_cpy = pruner.prune(model_cpy, individual)
            return (evaluator.evaluate(model_cpy),)

        tb.register("evaluate", evaluate)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=self._mutpb)
        tb.register("select", tools.selTournament, tournsize=self._tournsize)

        population = tb.population(n=self._pop_size)
        self._eval_pop(population, tb)

        for gen in range(self._n_gen):
            offspring = tb.select(population, len(population))
            offspring = list(map(tb.clone, offspring))

            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
        top10 = tools.selBest(population, k=10)

    def _eval_pop(self, population: Iterable[Any], toolbox: Toolbox) -> None:
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
