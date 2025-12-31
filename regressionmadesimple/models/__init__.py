"""
RegressionMadeSimple v3.0.0 - Models Module

This module provides a registry of regression models that can be used
with the new class-based API.

Example usage:
    >>> import regressionmadesimple as rms
    >>> 
    >>> # New v3.0.0 API (recommended)
    >>> model = rms.models.Linear(data, 'x', 'y')
    >>> 
    >>> # Or use with wrapper
    >>> model = rms.LinearRegressionModel.fit(data, 'x', 'y', model=rms.models.Linear)
"""

from .base import BaseModel
from .linear import Linear
from .quadratic import Quadratic
from .cubic import Cubic
from .curves import CustomCurve

__all__ = [
    'BaseModel',
    'Linear',
    'Quadratic',
    'Cubic',
    'CustomCurve',
]