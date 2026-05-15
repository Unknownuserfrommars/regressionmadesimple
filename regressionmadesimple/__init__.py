"""
RegressionMadeSimple v4.1.0

A minimalist machine learning toolkit that wraps scikit-learn for quick prototyping.
Just `import regressionmadesimple as rms` and go!

New in v4.1.0:
- **Experiment class**: full experiment workflow — smart column typing, multi-split
  management, scaler integration, model dedup, dill persistence.
- Legacy curves.py and base_class.py removed.

v4.0.0 highlights:
- Model Registry: Access models via rms.models.Linear, rms.models.Quadratic, etc.
- Enhanced BaseModel: Common functionality including save/load, scoring metrics (R², MAE, RMSE)
- Class-based model specification required (string names removed)

Example usage (Experiment):
    >>> import regressionmadesimple as rms
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> exp = rms.Experiment(df, target="y", out_path="./exp")
    >>> exp.fit_models({"lr": [LinearRegression()]})
    >>> exp.save()

Example usage (classic API):
    >>> import regressionmadesimple as rms
    >>> model = rms.models.Linear(data, 'x', 'y')
    >>> predictions = model.predict(new_data)
    >>> model.save_model('my_model.pkl')
"""

from . import models

# Experiment workflow (new in v4.1.0)
from .experiment import Experiment, _LOWER_IS_BETTER

# Backward compatibility: Keep old imports working
from .models.linear import Linear
from .models.quadratic import Quadratic
from .models.cubic import Cubic
from .models.curves import CustomCurve

# Import utilities
from .utils_preworks import Preworks, Logger
from .options import options, save_options, load_options, reset_options
from .wrapper import LinearRegressionModel

__version__ = "4.1.0-dev"

__all__ = [
    # Experiment workflow (new in v4.1.0)
    "Experiment",
    "_LOWER_IS_BETTER",
    # Models module (new in v3.0.0)
    "models",
    # Individual model classes (backward compatibility)
    "Linear",
    "Quadratic",
    "Cubic",
    "CustomCurve",
    # Utilities
    "Preworks",
    "Logger",
    "options",
    "save_options",
    "load_options",
    "reset_options",
    # Wrapper
    "LinearRegressionModel",
]
