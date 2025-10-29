# src/evolvepy/strategies/parent_selection.py

import random
from typing import List

from ..individual import Individual, GenotypeType


class TournamentSelection:
    """
    A parent selection strategy based on k-tournament.

    This callable object implements parent selection by repeatedly
    running tournaments. In each tournament, 'k' individuals are
    chosen randomly from the population, and the one with the
    best fitness "wins" and is selected as a parent.
    """
    
    def __init__(self, k: int = 3):
        """
        Initializes the tournament selection strategy.

        Args:
            k (int): The number of individuals to compete in each
                     tournament. A common value is 3. Must be > 0.
        """
        if k <= 0:
            raise ValueError("Tournament size 'k' must be greater than 0")
        
        # 'k' is the "state" of this strategy
        self.k = k

    def __call__(self, population: List[Individual[GenotypeType]]) -> List[Individual[GenotypeType]]:
        """
        Selects a list of parents from the population using k-tournament.

        This method selects a number of parents equal to the
        population size. Individuals can be (and often are)
        selected multiple times.

        Args:
            population: The current list of individuals to select from.
                        Assumes fitness has been evaluated.

        Returns:
            A new list of Individuals selected as parents.
        """
        parents = []
        pop_size = len(population)
        
        if pop_size == 0:
            return []
            
        for _ in range(pop_size):
            # Select k random contenders from the population
            # 'sample' selects *without* replacement (for a fair tournament)
            contenders = random.sample(population, self.k)
            
            # Find the winner (best fitness)
            # This assumes a maximization problem, as defined by the engine
            winner = max(contenders, key=lambda ind: ind.fitness)
            parents.append(winner)
            
        return parents


class UniformSelection:
    """
    A parent selection strategy that chooses parents uniformly at random.

    This is one of the simplest selection methods. It selects
    individuals with replacement, meaning a single individual can
    be chosen multiple times, and all individuals have an
    equal probability of being chosen.
    """
    
    # This strategy is "stateless", so __init__ is not needed.

    def __call__(self, population: List[Individual[GenotypeType]]) -> List[Individual[GenotypeType]]:
        """
        Selects a list of parents uniformly at random *with replacement*.

        Args:
            population: The current list of individuals to select from.

        Returns:
            A new list of Individuals selected as parents, with the
            same size as the original population.
        """
        pop_size = len(population)
        if pop_size == 0:
            return []
        
        # Select 'k' items from a list *with replacement*.
        parents = random.choices(population, k=pop_size)
        
        return parents