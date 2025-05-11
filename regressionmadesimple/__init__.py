from .linear import Linear
from .utils_preworks import Preworks, Logger
from .options import options, save_options, load_options, reset_options
from .wrapper import LinearRegressionModel
from .quadratic import Quadratic
from .cubic import Cubic
from .curves import CustomCurve

__version__ = "1.3.0"

__all__ = [
    "Linear",
    "Preworks",
    "options",
    "save_options",
    "load_options",
    "reset_options",
    "LinearRegressionModel",
    "Quadratic",
    "Cubic",
    "CustomCurve",
    'Logger',
]
