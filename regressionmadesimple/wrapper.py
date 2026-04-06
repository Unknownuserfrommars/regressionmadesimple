"""
Wrapper for fitting different regression types.

v4.0.0 removes legacy string-based model specification.
Use class-based model specification only.
"""

from typing import Type
from .models import Linear
from .models.base import BaseModel


class LinearRegressionModel:
    """
    Wrapper for fitting different regression types.

    v4.0.0 API style:
        model=rms.models.Linear
    """

    @staticmethod
    def fit(dataset, colX, colY, model: Type[BaseModel] = Linear, **kwargs):
        """
        Fit a regression model.

        Parameters:
            dataset: pd.DataFrame - The input dataset
            colX: str - The X column name (input feature)
            colY: str - The y column name (target variable)
            model: Model class - Regression model to use
                   (e.g., rms.models.Linear, rms.models.Quadratic, rms.models.Cubic)
            **kwargs: Additional arguments passed to model constructor
                     (e.g., testsize, randomstate, train_test_split)

        Returns:
            Fitted model instance
        """
        if isinstance(model, type) and issubclass(model, BaseModel):
            return model(dataset, colX, colY, **kwargs)

        raise TypeError(
            f"Invalid model type: {type(model)}. "
            "Expected a model class (e.g., rms.models.Linear). "
            "String-based model values were removed in v4.0.0."
        )
