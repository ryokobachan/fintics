"""Fintics Strategy Module

This module provides trading strategy classes and utilities.

Main Components:
    - Strategy: Base strategy class
    - OptimizeStrategy: Base class for optimizable strategies
    - Built-in Strategies: Pre-built strategies in the built_in submodule
"""

from .strategy import *
from .orderprice import *

# Import all built-in strategies
from .built_in import *

__all__ = ['Strategy', 'OptimizeStrategy']
