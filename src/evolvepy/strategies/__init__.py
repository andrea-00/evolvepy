# src/evolvepy/strategies/__init__.py

"""
The 'strategies' module contains all the composable "plugins"
for the EA engine.

This file flattens the namespace, allowing users to import
strategies like 'TournamentSelection' or 'OrderedCrossover'
directly from 'evolvepy.strategies' without needing to
know the internal file structure (e.g., 'parent_selection.py').
"""

# Import from crossover module
from .crossover import OrderedCrossover, CycleCrossover

# Import from mutation module
from .mutation import SwapMutation, InversionMutation
# (You could also add BitFlipMutation, GaussianMutation here if you keep them)

# Import from parent_selection module
from .parent_selection import TournamentSelection, UniformSelection

# Import from reproduction module
from .reproduction import StandardReproduction, MutationOnlyReproduction, ExclusiveReproduction

# Import from survivor_selection module
from .survivor_selection import PlusSelection, CommaSelection, PlusAgeBasedSelection



# Define the public API for 'from evolvepy.strategies import *'
__all__ = [
    # Crossover
    "OrderedCrossover", "CycleCrossover",
    # Mutation
    "SwapMutation", "InversionMutation",
    # Parent Selection
    "TournamentSelection", "UniformSelection",
    # Reproduction
    "StandardReproduction", "MutationOnlyReproduction", "ExclusiveReproduction",
    # Survivor Selection
    "PlusSelection", "CommaSelection", "PlusAgeBasedSelection",
]