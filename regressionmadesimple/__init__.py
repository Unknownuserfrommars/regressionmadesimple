"""
RegressionMadeSimple - A simple wrapper around scikit-learn for regression tasks.

Version: 2.0.0
Author: RegressionMadeSimple Team
"""

__version__ = "2.0.0"
__author__ = "RegressionMadeSimple Team"

# Import main API functions for easy access
from .api.fit import fit
from .models.linear import Linear
from .models.quadratic import Quadratic
from .models.cubic import Cubic

# Clean public API surface
__all__ = [
    "fit",
    "Linear", 
    "Quadratic",
    "Cubic",
    "__version__",
]