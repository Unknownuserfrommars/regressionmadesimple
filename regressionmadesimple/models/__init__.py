"""
Models module for RegressionMadeSimple.

Contains simple wrapper classes around scikit-learn regression models.
"""

from .linear import Linear
from .quadratic import Quadratic
from .cubic import Cubic

__all__ = ["Linear", "Quadratic", "Cubic"]