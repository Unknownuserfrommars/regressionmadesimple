"""
Validation utilities for RegressionMadeSimple.
"""

import numpy as np
from typing import Union, Tuple, Any, List
from sklearn.model_selection import train_test_split


def validate_input(X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert input data to numpy arrays.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Target values
        
    Returns
    -------
    X_validated : np.ndarray
        Validated feature array
    y_validated : np.ndarray
        Validated target array
        
    Raises
    ------
    ValueError
        If inputs have incompatible shapes or contain invalid values
    TypeError
        If inputs cannot be converted to numpy arrays
    """
    try:
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert input to numpy arrays: {e}")
    
    # Check dimensions
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got {y.ndim}D")
    
    # Check shapes match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. "
                        f"Got X: {X.shape[0]}, y: {y.shape[0]}")
    
    # Check for empty arrays
    if X.shape[0] == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    
    return X, y


def validate_split_ratio(split_ratio: List[Union[int, float]]) -> List[float]:
    """
    Validate and normalize split ratio parameter.
    
    Parameters
    ----------
    split_ratio : list of int or float
        Split ratio as [train_ratio, test_ratio], e.g., [8, 2] or [0.8, 0.2]
        
    Returns
    -------
    normalized_ratio : list of float
        Normalized split ratio where train_ratio + test_ratio = 1.0
        
    Raises
    ------
    ValueError
        If split_ratio is invalid
    """
    if not isinstance(split_ratio, (list, tuple)) or len(split_ratio) != 2:
        raise ValueError("split_ratio must be a list or tuple of exactly 2 numbers")
    
    train_ratio, test_ratio = split_ratio
    
    if not all(isinstance(x, (int, float)) and x > 0 for x in [train_ratio, test_ratio]):
        raise ValueError("split_ratio values must be positive numbers")
    
    # Normalize to sum to 1.0
    total = train_ratio + test_ratio
    normalized_train = train_ratio / total
    normalized_test = test_ratio / total
    
    return [normalized_train, normalized_test]


def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    split_ratio: List[Union[int, float]], 
    random_state: int = None,
    stratify: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets based on ratio.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target values
    split_ratio : list of int or float
        Split ratio as [train_ratio, test_ratio], e.g., [8, 2] means 80% train, 20% test
    random_state : int, optional
        Random state for reproducible splits
    stratify : np.ndarray, optional
        If not None, data is split in a stratified fashion using this as class labels
        
    Returns
    -------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    y_train : np.ndarray
        Training targets
    y_test : np.ndarray
        Testing targets
        
    Examples
    --------
    >>> X = np.random.randn(100, 2)
    >>> y = np.random.randn(100)
    >>> X_train, X_test, y_train, y_test = split_data(X, y, [8, 2])
    >>> print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    Train size: 80, Test size: 20
    """
    # Validate inputs
    X, y = validate_input(X, y)
    normalized_ratio = validate_split_ratio(split_ratio)
    
    # Calculate test size (sklearn uses test_size parameter)
    test_size = normalized_ratio[1]  # This is already normalized
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test


def validate_model_params(**params) -> dict:
    """
    Validate model parameters.
    
    Parameters
    ----------
    **params : dict
        Model parameters to validate
        
    Returns
    -------
    validated_params : dict
        Validated parameters
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    validated = {}
    
    for key, value in params.items():
        if key == 'fit_intercept':
            if not isinstance(value, bool):
                raise ValueError(f"fit_intercept must be boolean, got {type(value)}")
            validated[key] = value
            
        elif key == 'normalize':
            if not isinstance(value, bool):
                raise ValueError(f"normalize must be boolean, got {type(value)}")
            validated[key] = value
            
        elif key == 'include_bias':
            if not isinstance(value, bool):
                raise ValueError(f"include_bias must be boolean, got {type(value)}")
            validated[key] = value
            
        elif key == 'degree':
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"degree must be positive integer, got {value}")
            validated[key] = value
            
        elif key == 'split_ratio':
            if value is not None:
                validated[key] = validate_split_ratio(value)
            else:
                validated[key] = value
                
        elif key == 'random_state':
            if value is not None and not isinstance(value, int):
                raise ValueError(f"random_state must be integer or None, got {type(value)}")
            validated[key] = value
            
        else:
            # Pass through unknown parameters (for extensibility)
            validated[key] = value
    
    return validated


def check_is_fitted(model: Any) -> bool:
    """
    Check if a model has been fitted.
    
    Parameters
    ----------
    model : object
        Model instance to check
        
    Returns
    -------
    is_fitted : bool
        True if model is fitted, False otherwise
    """
    return hasattr(model, 'is_fitted') and model.is_fitted


def validate_prediction_input(X: Union[np.ndarray, list], n_features_expected: int = None) -> np.ndarray:
    """
    Validate input for prediction.
    
    Parameters
    ----------
    X : array-like
        Input features for prediction
    n_features_expected : int, optional
        Expected number of features
        
    Returns
    -------
    X_validated : np.ndarray
        Validated input array
        
    Raises
    ------
    ValueError
        If input has wrong shape or contains invalid values
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert input to numpy array: {e}")
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")
    
    if n_features_expected is not None and X.shape[1] != n_features_expected:
        raise ValueError(f"Expected {n_features_expected} features, got {X.shape[1]}")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    return X