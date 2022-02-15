import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch.nn as nn
from deap import base, creator, tools
from deap.base import Toolbox

from .constraint import Constraint
from .objective import Objective
from .pruner import Pruner


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def optimize(
        self, model: nn.Module, pruner: Pruner, objective: Objective, constraint: Constraint
    ) -> nn.Module:
        pass

    @abstractmethod
    def history(self) -> Any:
        pass


class GAOptimizer(Optimizer):
    def __init__(
        self,
        ind_len: int,
        pop_size: int,
        elite_num: int,
        tourn_size: int,
        n_gen: int,
        mutp: float,
        indp: float,
        verbose: bool = True,
        verbose_freq: int = 1,
    ) -> None:
        super().__init__()

        self._indl = ind_len
        self._pop_size = pop_size
        self._elite_num = elite_num
        self._tourn_size = tourn_size
        self._n_gen = n_gen
        self._mutp = mutp
        self._indp = indp
        self._verbose = verbose
        self._verbose_freq = verbose_freq
        self._history = None

    def optimize(
        self, model: nn.Module, pruner: Pruner, objective: Objective, constraint: Constraint
    ) -> nn.Module:
        self.__pruner = pruner
        self.__obj = objective
        self.__const = constraint
        self.__model = model

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()

        tb.register("attr_bool", random.randint, 0, 1)
        tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bool, n=self._indl)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=self._indp)
        tb.register("select", tools.selTournament, tournsize=self._tourn_size)

        logbook = tools.Logbook()
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population = self._generate_pop(pop_size=self._pop_size, toolbox=tb)
        self._evaluate_pop(population)
        best = tools.selBest(population, k=1)

        for gen in range(self._n_gen):
            new_pop = list(map(tb.clone, self._elite_set(population)))

            while len(new_pop) < len(population):
                off1, off2 = self._crossover(population, tb)
                off1, off2 = self._mutation(off1, tb), self._mutation(off2, tb)

                if self._feasible(off1):
                    new_pop.append(off1)

                if self._feasible(off2):
                    new_pop.append(off2)

            population = new_pop
            self._evaluate_pop(population)
            best = self._keep_best(best, population)

            record = stats.compile(population)
            logbook.record(gen=gen, **record)

            if self._verbose and gen % self._verbose_freq == 0:
                stats = ", ".join([f"{k.capitalize()} = {v:.4f}" for k, v in record.items()])
                print(f"Generation {gen:04d}: {stats}")

        self._history = logbook
        model_cpy = deepcopy(model)
        return pruner.prune(model_cpy, best)

    def history(self) -> Any:
        return self._history

    def _generate_pop(self, pop_size: int, toolbox: Toolbox) -> Iterable[Any]:
        pop = []

        while len(pop) < pop_size:
            candidate = toolbox.individual()
            if self._feasible(candidate):
                pop.append(candidate)

        return pop

    def _evaluate_pop(self, population: Iterable[Any]) -> None:
        for individual in population:
            individual.fitness.values = self._evaluate(individual)

    def _feasible(self, individual: Any) -> bool:
        model = self._create_model(individual)
        feasible = self.__const.feasible(model) if self.__const is not None else True
        del model
        return feasible

    def _evaluate(self, individual: Any) -> Tuple[float, ...]:
        model = self._create_model(individual)
        fitness = self.__obj.evaluate(model)
        del model
        return fitness

    def _create_model(self, individual: Any) -> nn.Module:
        model_cpy = deepcopy(self.__model)
        model_cpy = self.__pruner.prune(model_cpy, individual)
        return model_cpy

    def _crossover(self, population: Iterable[Any], toolbox: Toolbox) -> Tuple[Any]:
        # Parent selection
        p1, p2 = toolbox.select(population, 2)
        off1, off2 = toolbox.clone(p1), toolbox.clone(p2)

        # Crossover
        off1, off2 = toolbox.mate(off1, off2)
        del off1.fitness.values
        del off2.fitness.values

        return (off1, off2)

    def _mutation(self, individual: Any, toolbox: Toolbox) -> Any:
        if random.random() <= self._mutp:
            individual = toolbox.mutate(individual)[0]
            del individual.fitness.values
        return individual

    def _elite_set(self, population: Iterable[Any]) -> List[Any]:
        return tools.selBest(population, k=self._elite_num)

    def _keep_best(self, curr_best: Any, population: Iterable[Any]) -> Any:
        return tools.selBest([curr_best] + tools.selBest(population, k=1), k=1)
