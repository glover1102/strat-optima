"""
QTAlgo Super26 Strategy - Walk-Forward Optimization Framework

A comprehensive Python framework for walk-forward optimization of the QTAlgo Super26 
trading strategy with production-ready deployment capabilities.
"""

from .strategy import *
from .data import *
from .optimization import *
from .utils import *

__version__ = '1.0.0'
__author__ = 'QTAlgo Team'

__all__ = [
    'strategy',
    'data',
    'optimization',
    'utils'
]
