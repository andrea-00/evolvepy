# src/evolvepy/__init__.py

"""
evolvepy: A Framework for Evolutionary Algorithms in Python.

This package provides the core classes for building,
running, and analyzing EA experiments.
"""

# Expose the core engine and individual classes
from .engine import EvolutionaryAlgorithm
from .individual import Individual

# Expose the primary logger classes and enums
from .logger import BaseLogger, Logger, LogLevel

# Expose the problem blueprint
from .base_problem import BaseProblem

# Define what 'from evolvepy import *' will import
# This is the "Public API" of the top-level package.
__all__ = [
    "EvolutionaryAlgorithm",
    "Individual",
    "BaseLogger",
    "Logger",
    "LogLevel",
    "BaseProblem"
]