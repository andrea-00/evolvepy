# src/evolvepy/analysis/__init__.py

"""
The 'analysis' module provides helper functions for
post-run analysis and visualization of EA results.
"""

# Import all public plotting functions
from .plotting import plot_convergence

# Define the public API
__all__ = [
    "plot_convergence",
]