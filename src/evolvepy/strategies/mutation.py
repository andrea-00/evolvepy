# src/evolvepy/strategies/mutation.py

import random
from typing import List, Union, Callable, Any

from ..individual import Individual, GenotypeType


class SwapMutation:
    """
    Performs mutation by swapping two randomly selected genes in a genotype.

    This strategy is designed for **permutation-based representations**,
    where the genotype is a sequence (e.g., a list) and the order
    of elements is critical. 
    
    It guarantees that the resulting genotype remains a valid permutation
    of the original elements.
    """

    def __init__(self, 
                 individual_mutation_prob: Union[float, Callable[[int], float]]):
        """
        Initializes the swap mutation strategy.

        Args:
            individual_mutation_prob: The probability (0.0 to 1.0) that
                any single *individual* (genotype) will undergo one
                swap operation.
                This can be a fixed float or a callable scheduler (fn(gen) -> float).
        """
        self.prob_schedule = individual_mutation_prob

    def __call__(self, 
                 offspring: List[Individual[List[Any]]], 
                 current_generation: int) -> List[Individual[List[Any]]]:
        """
        Applies swap mutation to a list of offspring.

        Args:
            offspring: The list of offspring Individuals to mutate.
                       Expects GenotypeType to be a List.
            current_generation: The current generation number (for scheduling).

        Returns:
            The same list of offspring, with genotypes potentially modified.
            
        Raises:
            TypeError: If an individual's genotype is not a 'list'.
        """
        
        # Resolve the probability for this generation
        prob = self.prob_schedule(current_generation) if callable(self.prob_schedule) else self.prob_schedule

        for ind in offspring:
            # Check if this individual should be mutated
            if random.random() < prob:
                if not isinstance(ind.genotype, list):
                    raise TypeError(f"SwapMutation requires 'list' genotypes, "
                                    f"got {type(ind.genotype)}")
                
                gen_len = len(ind.genotype)
                if gen_len < 2:
                    # Cannot swap a list of length 0 or 1
                    continue 

                # 1. Select two *distinct* random indices
                idx1, idx2 = random.sample(range(gen_len), 2)

                # 2. Perform the swap (in-place)
                temp = ind.genotype[idx1]
                ind.genotype[idx1] = ind.genotype[idx2]
                ind.genotype[idx2] = temp
        
        return offspring


class InversionMutation:
    """
    Performs mutation by inverting a randomly selected subsequence
    of the genotype.

    Also known as Reverse Sequence Mutation (RSM). This strategy is
    highly effective for **permutation-based representations**.
    It preserves the set of elements in the genotype, changing
    only their order within the selected subsequence.
    
    Example: [A, B, |C, D, E,| F] -> [A, B, |E, D, C,| F]
    """
    
    def __init__(self, 
                 individual_mutation_prob: Union[float, Callable[[int], float]]):
        """
        Initializes the inversion mutation strategy.

        Args:
            individual_mutation_prob: The probability (0.0 to 1.0) that
                any single *individual* will undergo one inversion.
                This can be a fixed float or a callable scheduler (fn(gen) -> float).
        """
        self.prob_schedule = individual_mutation_prob

    def __call__(self, 
                 offspring: List[Individual[List[Any]]], 
                 current_generation: int) -> List[Individual[List[Any]]]:
        """
        Applies inversion mutation to a list of offspring.

        Args:
            offspring: The list of offspring Individuals to mutate.
                       Expects GenotypeType to be a List.
            current_generation: The current generation number (for scheduling).

        Returns:
            The same list of offspring, with genotypes potentially modified.
            
        Raises:
            TypeError: If an individual's genotype is not a 'list'.
        """
        
        # Resolve the probability for this generation
        prob = self.prob_schedule(current_generation) if callable(self.prob_schedule) else self.prob_schedule

        for ind in offspring:
            # Check if this individual should be mutated
            if random.random() < prob:
                if not isinstance(ind.genotype, list):
                    raise TypeError(f"InversionMutation requires 'list' genotypes, "
                                    f"got {type(ind.genotype)}")
                
                gen_len = len(ind.genotype)
                if gen_len < 2:
                    # Cannot invert a list of length 0 or 1
                    continue 

                # 1. Select two *distinct* random indices
                idx1, idx2 = random.sample(range(gen_len), 2)

                # 2. Ensure idx1 is the smaller one (start point)
                start = min(idx1, idx2)
                end = max(idx1, idx2) # End point (inclusive)
                
                # 3. Perform the inversion (in-place)
                # This reverses the sub-list from 'start' to 'end'
                sub_list = ind.genotype[start:end+1]
                sub_list.reverse()
                ind.genotype[start:end+1] = sub_list
        
        return offspring