import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, List, Tuple

import numpy as np
from deap import creator, tools
from deap.base import Toolbox, Fitness

from .constraint import Constraint
from .objective import Objective


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def optimize(self, objective: Objective, constraint: Constraint) -> Any:
        pass


class GAOptimizer(Optimizer):
    def __init__(
        self,
        ind_size: int,
        pop_size: int,
        elite_num: int,
        tourn_size: int,
        n_gen: int,
        mutp: float,
        mut_indp: float,
        cx_indp: float,
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
        verbose_freq: int = 1,
    ) -> None:
        super().__init__()

        self._ind_size = ind_size
        self._pop_size = pop_size
        self._elite_num = elite_num
        self._tourn_size = tourn_size
        self._n_gen = n_gen
        self._mutp = mutp
        self._mut_indp = mut_indp
        self._cx_indp = cx_indp
        self._init_pop = init_pop
        self._verbose = verbose
        self._verbose_freq = verbose_freq

        self._best = None
        self._population = None
        self._toolbox = None
        self._history = None

    def optimize(self, objective: Objective, constraint: Constraint) -> Any:
        self.__obj = objective
        self.__constr = constraint

        creator.create("FitnessMax", Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self._toolbox = self._create_toolbox()
        self._history = tools.Logbook()

        self._population = (
            self._generate_pop(creator.Individual)
            if self._init_pop is None
            else self._initialize_pop(creator.Individual)
        )
        self._handle_generation(gen_num=0)

        for gen in range(1, self._n_gen + 1):
            new_pop = list(map(self._toolbox.clone, self._elite_set(self._population)))

            while len(new_pop) < len(self._population):
                off1, off2 = self._crossover(self._population)
                off1, off2 = self._mutation(off1), self._mutation(off2)

                if constraint.feasible(off1):
                    new_pop.append(off1)

                if constraint.feasible(off2):
                    new_pop.append(off2)

            self._population = new_pop
            self._handle_generation(gen_num=gen)

        return self._best

    def best(self) -> Any:
        return self._best

    def population(self) -> Iterable[Any]:
        return self._population

    def history(self) -> Any:
        return self._history

    def _handle_generation(self, gen_num: int) -> None:
        # Evaluate population
        for individual in self._population:
            individual.fitness.values = self.__obj.evaluate(individual)

        # Keep current best found solution
        self._best = (
            self._keep_best(self._best, self._population)
            if self._best is not None
            else tools.selBest(self._population, k=1)[0]
        )

        # Compute statistics of current population
        stats = self._create_stats()
        record = stats.compile(self._population)
        self._history.record(gen=gen_num, best=self._best, **record)

        # Print statistcs to terminal
        if self._verbose and gen_num % self._verbose_freq == 0:
            stats_str = ", ".join([f"{k.capitalize()} = {v:.4f}" for k, v in record.items()])
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Generation {gen_num:04d}: {stats_str}")

    def _generate_pop(self, ind_cls: Any) -> Iterable[Any]:
        pop = []

        while len(pop) < self._pop_size:
            for i in range(self._pop_size):
                p = (i + 1) / self._pop_size
                ind = ind_cls([random.random() <= p for _ in range(self._ind_size)])

                if self.__constr.feasible(ind):
                    pop.append(ind)

                if len(pop) == self._pop_size:
                    break

        return pop

    def _initialize_pop(self, ind_cls: Any) -> Iterable[Any]:
        return [ind_cls(content) for content in self._init_pop]

    def _crossover(self, population: Iterable[Any]) -> Tuple[Any]:
        # Parent selection
        p1, p2 = self._toolbox.select(population, 2)
        off1, off2 = self._toolbox.clone(p1), self._toolbox.clone(p2)

        # Crossover
        off1, off2 = self._toolbox.mate(off1, off2)
        del off1.fitness.values
        del off2.fitness.values

        return (off1, off2)

    def _mutation(self, individual: Any) -> Any:
        if random.random() <= self._mutp:
            individual = self._toolbox.mutate(individual)[0]
            del individual.fitness.values

        return individual

    def _elite_set(self, population: Iterable[Any]) -> List[Any]:
        return tools.selBest(population, k=self._elite_num)

    def _keep_best(self, curr_best: Any, population: Iterable[Any]) -> Any:
        return tools.selBest([curr_best] + tools.selBest(population, k=1), k=1)[0]

    def _create_toolbox(self) -> Toolbox:
        tb = Toolbox()

        tb.register("attr_bool", random.randint, 0, 1)
        tb.register("mate", tools.cxUniform, indpb=self._cx_indp)
        tb.register("mutate", tools.mutFlipBit, indpb=self._mut_indp)
        tb.register("select", tools.selTournament, tournsize=self._tourn_size)

        return tb

    def _create_stats(self) -> tools.Statistics:
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats
