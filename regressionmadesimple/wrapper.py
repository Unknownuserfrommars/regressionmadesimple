"""
Wrapper for fitting different regression types - Updated for v3.0.0
Supports both class-based (new) and string-based (legacy) model specification
"""

import warnings
from typing import Union, Type
from .models import Linear, Quadratic, Cubic
from .models.base import BaseModel


class LinearRegressionModel:
    """
    Wrapper for fitting different regression types.
    
    Supports two API styles:
    1. Class-based (v3.0.0+, recommended): model=rms.models.Linear
    2. String-based (legacy, deprecated): model='linear'
    """

    @staticmethod
    def fit(dataset, colX, colY, model: Union[str, Type[BaseModel]] = 'linear', **kwargs):
        """
        Fit a regression model.
        
        Parameters:
            dataset: pd.DataFrame - The input dataset
            colX: str - The X column name (input feature)
            colY: str - The y column name (target variable)
            model: str or Model class - Regression model to use
                   - Class-based (recommended): rms.models.Linear, rms.models.Quadratic, rms.models.Cubic
                   - String-based (deprecated): 'linear', 'quadratic', 'cubic'
            **kwargs: Additional arguments passed to model constructor
                     (e.g., testsize, randomstate, train_test_split)
        
        Returns:
            Fitted model instance
        
        Examples:
            >>> import regressionmadesimple as rms
            >>> 
            >>> # New v3.0.0 API (recommended)
            >>> model = rms.LinearRegressionModel.fit(
            ...     data, 'x', 'y', 
            ...     model=rms.models.Linear
            ... )
            >>> 
            >>> # Legacy API (deprecated, will show warning)
            >>> model = rms.LinearRegressionModel.fit(
            ...     data, 'x', 'y', 
            ...     model='linear'
            ... )
        """
        # Handle class-based model (new API)
        if isinstance(model, type) and issubclass(model, BaseModel):
            return model(dataset, colX, colY, **kwargs)
        
        # Handle string-based model (legacy API)
        elif isinstance(model, str):
            # Issue deprecation warning
            warnings.warn(
                f"String-based model specification (model='{model}') is deprecated and will be "
                f"removed in v4.0.0. Please use class-based specification instead:\n"
                f"  model=rms.models.{model.capitalize()}\n"
                f"Example: rms.LinearRegressionModel.fit(data, 'x', 'y', model=rms.models.Linear)",
                DeprecationWarning,
                stacklevel=2
            )
            
            # Map string to class
            model_name = model.strip().lower().capitalize()
            model_map = {
                'Linear': Linear,
                'Quadratic': Quadratic,
                'Cubic': Cubic,
            }
            
            model_class = model_map.get(model_name)
            if not model_class:
                raise ValueError(
                    f"Model '{model}' not supported. "
                    f"Available options: {', '.join(model_map.keys())}"
                )
            
            return model_class(dataset, colX, colY, **kwargs)
        
        else:
            raise TypeError(
                f"Invalid model type: {type(model)}. "
                f"Expected a model class (e.g., rms.models.Linear) or string (deprecated)."
            )