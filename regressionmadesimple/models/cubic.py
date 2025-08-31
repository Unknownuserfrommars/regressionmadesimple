"""
Cubic regression model wrapper.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from typing import Union, Optional, List
from ..utils.validation import validate_input, split_data


class Cubic:
    """
    Simple wrapper for cubic (polynomial degree 3) regression.
    
    Uses scikit-learn's PolynomialFeatures with LinearRegression
    to create cubic models.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    include_bias : bool, default=True
        Whether to include bias column in polynomial features
    split_ratio : list of int or float, optional
        Split ratio as [train_ratio, test_ratio], e.g., [8, 2] means 80% train, 20% test.
        If provided, data will be automatically split during fit().
    random_state : int, optional
        Random state for reproducible data splits
    
    Attributes
    ----------
    model : sklearn.pipeline.Pipeline
        Pipeline containing PolynomialFeatures and LinearRegression
    is_fitted : bool
        Whether the model has been fitted
    X_train : np.ndarray, optional
        Training features (available if split_ratio was used)
    y_train : np.ndarray, optional
        Training targets (available if split_ratio was used)
    X_test : np.ndarray, optional
        Testing features (available if split_ratio was used)
    y_test : np.ndarray, optional
        Testing targets (available if split_ratio was used)
    """
    
    def __init__(
        self, 
        fit_intercept: bool = True, 
        include_bias: bool = True,
        split_ratio: Optional[List[Union[int, float]]] = None,
        random_state: Optional[int] = None
    ):
        self.fit_intercept = fit_intercept
        self.include_bias = include_bias
        self.split_ratio = split_ratio
        self.random_state = random_state
        
        # Create pipeline with polynomial features (degree=3) and linear regression
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=include_bias)),
            ('linear', LinearRegression(fit_intercept=fit_intercept))
        ])
        self.is_fitted = False
        
        # Test data attributes (will be set if split_ratio is used)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'Cubic':
        """
        Fit the cubic regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data features
        y : array-like, shape (n_samples,)
            Training data targets
            
        Returns
        -------
        self : Cubic
            Returns self for method chaining
        """
        X, y = validate_input(X, y)
        
        # Handle data splitting if split_ratio is provided
        if self.split_ratio is not None:
            X_train, X_test, y_train, y_test = split_data(
                X, y, self.split_ratio, self.random_state
            )
            # Store all data as attributes
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        else:
            X_train, y_train = X, y
            self.X_train = X_train
            self.y_train = y_train
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        return self.model.predict(X)
    
    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Calculate R² score for the model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            True target values
            
        Returns
        -------
        score : float
            R² score
        """
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_test_score(self) -> Optional[float]:
        """
        Get R² score on test data (if available).
        
        Returns
        -------
        test_score : float or None
            R² score on test data, or None if no test data available
        """
        if self.X_test is not None and self.y_test is not None:
            return self.score(self.X_test, self.y_test)
        return None
    
    def get_train_score(self) -> Optional[float]:
        """
        Get R² score on training data.
        
        Returns
        -------
        train_score : float or None
            R² score on training data, or None if not fitted
        """
        if self.X_train is not None and self.y_train is not None:
            return self.score(self.X_train, self.y_train)
        return None
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = {
            "fit_intercept": self.fit_intercept,
            "include_bias": self.include_bias,
            "degree": 3,
            "split_ratio": self.split_ratio,
            "random_state": self.random_state
        }
        
        if self.is_fitted:
            linear_model = self.model.named_steps['linear']
            params.update({
                "coefficients": linear_model.coef_,
                "intercept": linear_model.intercept_,
                "n_polynomial_features": len(linear_model.coef_)
            })
            
            if self.split_ratio is not None:
                params.update({
                    "n_samples_train": len(self.y_train) if self.y_train is not None else None,
                    "n_samples_test": len(self.y_test) if self.y_test is not None else None,
                    "train_r2_score": self.get_train_score(),
                    "test_r2_score": self.get_test_score()
                })
        
        return params
    
    def get_feature_names(self, input_features: list = None) -> list:
        """
        Get names of polynomial features.
        
        Parameters
        ----------
        input_features : list, optional
            Names of input features
            
        Returns
        -------
        feature_names : list
            Names of polynomial features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature names")
        
        poly_transformer = self.model.named_steps['poly']
        return poly_transformer.get_feature_names_out(input_features)
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        split_info = f", split_ratio={self.split_ratio}" if self.split_ratio else ""
        return f"Cubic(fit_intercept={self.fit_intercept}, include_bias={self.include_bias}{split_info}) - {status}"