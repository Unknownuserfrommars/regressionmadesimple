"""
Custom Curve Fitting Model - Refactored for v3.0.0
Supports custom basis functions for flexible curve fitting
"""

from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import BaseModel


class CustomCurve(BaseModel):
    """
    Custom curve fitting using user-defined basis functions.
    Inherits common methods from BaseModel.
    
    Note: Still in development. Use with caution.
    """
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        colX: str,
        colY: str,
        basis_funcs: Optional[List[str]] = None,
        testsize: float = 0.15,
        randomstate: int = 1,
        train_test_split: bool = True,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Custom Curve model with basis functions.
        
        Args:
            dataset: The dataset to input
            colX: the X column (input)
            colY: the y column (output)
            basis_funcs: List of basis function names (e.g., ["x", "x^2", "cos(x)"])
            testsize: The size of the train_test_split, default 0.15
            randomstate: The random state of train_test_split, default 1
            train_test_split: Whether to split the dataset. Defaults to True
            X_train, y_train, X_test, y_test: Pre-split data (optional)
        """
        # Default basis functions
        if basis_funcs is None:
            basis_funcs = ["x", "cos(x)", "sin(x)"]
        self.basis_funcs = basis_funcs
        
        # Initialize base class
        super().__init__(
            dataset=dataset,
            colX=colX,
            colY=colY,
            testsize=testsize,
            randomstate=randomstate,
            train_test_split=train_test_split,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # Transform features and fit
        self.fit()
    
    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features using basis functions.
        
        Args:
            X: Input array of shape (n_samples, 1)
            
        Returns:
            Transformed features array
        """
        features = []
        for func in self.basis_funcs:
            if func == "x":
                features.append(X)
            elif func == "x^2":
                features.append(X ** 2)
            elif func == "x^3":
                features.append(X ** 3)
            elif func == "cos(x)":
                features.append(np.cos(X))
            elif func == "sin(x)":
                features.append(np.sin(X))
            elif func == "log(x)":
                features.append(np.log(X + 1e-5))  # Avoid log(0)
            elif func == "exp(x)":
                features.append(np.exp(np.clip(X, -10, 10)))  # Prevent overflow
            elif func == "sqrt(x)":
                features.append(np.sqrt(np.abs(X)))
        return np.hstack(features)
    
    def fit(self):
        """
        Fit the custom curve model.
        """
        # Transform training data
        X_train_transformed = self._transform_features(self.X_train.values)
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_train_transformed, self.y_train)
        
        # Compute metrics if test data is available
        self._compute_metrics()
    
    def predict(self, X_new: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_new: New input data
            
        Returns:
            Predictions as numpy array
        """
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values
        X_transformed = self._transform_features(X_new)
        return self.model.predict(X_transformed)
    
    def summary(self):
        """
        Get model summary including basis functions and metrics.
        
        Returns:
            Dictionary with model information
        """
        base_summary = super().summary()
        base_summary.update({
            'model': 'Custom Curve Fitting',
            'basis_functions': self.basis_funcs,
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
        })
        return base_summary