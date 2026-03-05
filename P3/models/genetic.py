"""Genetic Algorithm baseline for resource management."""

from __future__ import annotations
import numpy as np


class GeneticRM:
    def __init__(self, pop_size=30, action_dim=4, mutation_rate=0.15):
        self.pop_size = pop_size
        self.action_dim = action_dim
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(-1, 1, (pop_size, action_dim))
        self.fitnesses = np.zeros(pop_size)
        self._idx = 0

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(self.population[self._idx % self.pop_size], -1, 1)

    def store(self, *args):
        pass

    def set_fitness(self, fitness: float):
        self.fitnesses[self._idx % self.pop_size] = fitness
        self._idx += 1

    def evolve(self):
        new_pop = np.zeros_like(self.population)
        for i in range(self.pop_size):
            a, b = np.random.randint(0, self.pop_size, 2)
            winner = a if self.fitnesses[a] >= self.fitnesses[b] else b
            new_pop[i] = self.population[winner].copy()
        for i in range(0, self.pop_size - 1, 2):
            if np.random.random() < 0.8:
                pt = np.random.randint(1, self.action_dim)
                new_pop[i, pt:], new_pop[i+1, pt:] = (
                    new_pop[i+1, pt:].copy(), new_pop[i, pt:].copy())
        mask = np.random.random(new_pop.shape) < self.mutation_rate
        new_pop[mask] = np.random.uniform(-1, 1, mask.sum())
        self.population = new_pop
        self.fitnesses = np.zeros(self.pop_size)
        self._idx = 0

    def update(self):
        return 0.0
