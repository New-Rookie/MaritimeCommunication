"""
Genetic Algorithm (GA) for neighbour discovery parameter optimisation.

Population of parameter configurations evolves through selection,
crossover, and mutation.  Fitness = accuracy - α·energy.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


class GeneticOptimizer:

    POWER_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    SCAN_DURATIONS = [0.05, 0.1, 0.2, 0.5]
    VERIFY_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]

    def __init__(
        self,
        pop_size: int = 30,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        alpha_energy: float = 0.01,
    ):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.alpha = alpha_energy

        self.population = self._init_population()
        self.fitnesses = np.zeros(pop_size)

    def _init_population(self) -> np.ndarray:
        """Each individual: [power_idx, scan_idx, thresh_idx]."""
        pop = np.zeros((self.pop_size, 3), dtype=int)
        for i in range(self.pop_size):
            pop[i, 0] = np.random.randint(len(self.POWER_LEVELS))
            pop[i, 1] = np.random.randint(len(self.SCAN_DURATIONS))
            pop[i, 2] = np.random.randint(len(self.VERIFY_THRESHOLDS))
        return pop

    def get_params(self, idx: int) -> Tuple[float, float, float]:
        g = self.population[idx]
        return (self.POWER_LEVELS[g[0]], self.SCAN_DURATIONS[g[1]],
                self.VERIFY_THRESHOLDS[g[2]])

    def get_best_params(self) -> Tuple[float, float, float]:
        best = int(np.argmax(self.fitnesses))
        return self.get_params(best)

    def set_fitness(self, idx: int, accuracy: float, energy: float):
        self.fitnesses[idx] = accuracy - self.alpha * energy

    def evolve(self):
        # Tournament selection
        new_pop = np.zeros_like(self.population)
        for i in range(self.pop_size):
            a, b = np.random.randint(0, self.pop_size, size=2)
            winner = a if self.fitnesses[a] >= self.fitnesses[b] else b
            new_pop[i] = self.population[winner].copy()

        # Crossover
        for i in range(0, self.pop_size - 1, 2):
            if np.random.random() < self.crossover_rate:
                pt = np.random.randint(1, 3)
                new_pop[i, pt:], new_pop[i + 1, pt:] = (
                    new_pop[i + 1, pt:].copy(), new_pop[i, pt:].copy()
                )

        # Mutation
        limits = [len(self.POWER_LEVELS), len(self.SCAN_DURATIONS),
                  len(self.VERIFY_THRESHOLDS)]
        for i in range(self.pop_size):
            for j in range(3):
                if np.random.random() < self.mutation_rate:
                    new_pop[i, j] = np.random.randint(limits[j])

        self.population = new_pop
        self.fitnesses = np.zeros(self.pop_size)
