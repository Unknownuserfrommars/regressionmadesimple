"""
Function-style API for RegressionMadeSimple.

Provides a simple fit() function that acts as a one-liner interface
for v4.0.0 class-based regression models.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any, List, Type
from sklearn.metrics import r2_score
from ..models.base import BaseModel
from ..models.linear import Linear
from ..utils.validation import validate_input, split_data


def _build_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, str, str]:
    if X.shape[1] != 1:
        raise ValueError(
            "RMS v4 function API currently supports single-feature regression only. "
            f"Got X with {X.shape[1]} features."
        )

    col_x = "x"
    col_y = "y"
    dataset = pd.DataFrame({col_x: X[:, 0], col_y: y})
    return dataset, col_x, col_y


def fit(
    X: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    model: Type[BaseModel] = Linear,
    split_ratio: Optional[List[Union[int, float]]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[Any, dict]:
    """
    Fit a regression model with a simple one-liner API.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data features
    y : array-like, shape (n_samples,)
        Training data targets
    model : BaseModel subclass, default=Linear
        Model class to use. Options: Linear, Quadratic, Cubic, or CustomCurve.
    split_ratio : list of int or float, optional
        Split ratio as [train_ratio, test_ratio], e.g., [8, 2] means 80% train, 20% test.
        If None, uses all data for training.
    random_state : int, optional
        Random state for reproducible data splits
    **kwargs : dict
        Additional parameters to pass to the model constructor

    Returns
    -------
    fitted_model : object
        The fitted model instance
    results : dict
        Dictionary containing fit results and metrics
    """
    # Validate inputs
    X, y = validate_input(X, y)

    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(
            f"model must be a BaseModel subclass, got {type(model)}. "
            "String-based model values were removed in v4.0.0."
        )

    dataset, col_x, col_y = _build_dataset(X, y)

    # Handle data splitting
    if split_ratio is not None:
        X_train, X_test, y_train, y_test = split_data(X, y, split_ratio, random_state)

        train_df = pd.DataFrame({col_x: X_train[:, 0]})
        test_df = pd.DataFrame({col_x: X_test[:, 0]})
        y_train_df = pd.DataFrame({col_y: y_train})
        y_test_df = pd.DataFrame({col_y: y_test})

        fitted_model = model(
            dataset,
            col_x,
            col_y,
            train_test_split=False,
            X_train=train_df,
            y_train=y_train_df,
            X_test=test_df,
            y_test=y_test_df,
            **kwargs,
        )

        train_predictions = fitted_model.predict(train_df).flatten()
        test_predictions = fitted_model.predict(test_df).flatten()

        results = {
            "model_type": model.__name__,
            "n_features": X.shape[1],
            "n_samples_total": X.shape[0],
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "split_ratio": split_ratio,
            "random_state": random_state,
            "train_r2_score": r2_score(y_train, train_predictions),
            "test_r2_score": r2_score(y_test, test_predictions),
            "r2_score": r2_score(y_test, test_predictions),
            "summary": fitted_model.summary(),
        }
        return fitted_model, results

    fitted_model = model(dataset, col_x, col_y, train_test_split=False, **kwargs)
    train_predictions = fitted_model.predict(fitted_model.X_train).flatten()

    results = {
        "model_type": model.__name__,
        "n_features": X.shape[1],
        "n_samples_total": X.shape[0],
        "n_samples_train": X.shape[0],
        "r2_score": r2_score(y, train_predictions),
        "summary": fitted_model.summary(),
    }

    return fitted_model, results
