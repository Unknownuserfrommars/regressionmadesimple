"""
RegressionMadeSimple v3.0.0

A minimalist machine learning toolkit that wraps scikit-learn for quick prototyping.
Just `import regressionmadesimple as rms` and go!

New in v3.0.0:
- Model Registry: Access models via rms.models.Linear, rms.models.Quadratic, etc.
- Enhanced BaseModel: Common functionality including save/load, scoring metrics (R², MAE, RMSE)
- Improved API: Class-based model specification (recommended) with backward compatibility
- Better code organization: Models in dedicated submodule

Example usage:
    >>> import regressionmadesimple as rms
    >>> import pandas as pd
    >>> 
    >>> # Load your data
    >>> data = pd.read_csv('data.csv')
    >>> 
    >>> # New v3.0.0 API (recommended)
    >>> model = rms.models.Linear(data, 'x', 'y')
    >>> predictions = model.predict(new_data)
    >>> model.save_model('my_model.pkl')
    >>> 
    >>> # Or use with wrapper
    >>> model = rms.LinearRegressionModel.fit(data, 'x', 'y', model=rms.models.Linear)
    >>> 
    >>> # Legacy API (still supported with deprecation warning)
    >>> model = rms.Linear(data, 'x', 'y')
"""

# Import models module (new in v3.0.0)
from . import models

# Backward compatibility: Keep old imports working
from .models.linear import Linear
from .models.quadratic import Quadratic
from .models.cubic import Cubic
from .models.curves import CustomCurve

# Import utilities
from .utils_preworks import Preworks, Logger
from .options import options, save_options, load_options, reset_options
from .wrapper import LinearRegressionModel

__version__ = "3.0.0"

__all__ = [
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