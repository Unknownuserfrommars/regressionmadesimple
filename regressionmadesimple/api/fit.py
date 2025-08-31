"""
Function-style API for RegressionMadeSimple.

Provides a simple fit() function that acts as a one-liner interface
to various regression models.
"""

import numpy as np
from typing import Union, Tuple, Optional, Any, List
from ..models.linear import Linear
from ..models.quadratic import Quadratic
from ..models.cubic import Cubic
from ..utils.validation import validate_input, split_data


def fit(
    X: Union[np.ndarray, list], 
    y: Union[np.ndarray, list], 
    model: str = "Linear",
    split_ratio: Optional[List[Union[int, float]]] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[Any, dict]:
    """
    Fit a regression model with a simple one-liner API.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data features
    y : array-like, shape (n_samples,)
        Training data targets
    model : str, default="Linear"
        Model type to use. Options: "Linear", "Quadratic", "Cubic"
    split_ratio : list of int or float, optional
        Split ratio as [train_ratio, test_ratio], e.g., [8, 2] means 80% train, 20% test.
        If None, uses all data for training.
    random_state : int, optional
        Random state for reproducible data splits
    **kwargs : dict
        Additional parameters to pass to the model
        
    Returns
    -------
    fitted_model : object
        The fitted model instance with test data accessible as attributes if split_ratio is provided
    results : dict
        Dictionary containing fit results and metrics
        
    Examples
    --------
    >>> import numpy as np
    >>> from regressionmadesimple import fit
    >>> 
    >>> # Generate sample data
    >>> X = np.random.randn(100, 2)
    >>> y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    >>> 
    >>> # Fit a linear model without data splitting
    >>> model, results = fit(X, y, model="Linear")
    >>> print(f"R² Score: {results['r2_score']:.3f}")
    >>> 
    >>> # Fit a linear model with 80-20 train-test split
    >>> model, results = fit(X, y, model="Linear", split_ratio=[8, 2])
    >>> print(f"Train R²: {results['train_r2_score']:.3f}")
    >>> print(f"Test R²: {results['test_r2_score']:.3f}")
    >>> print(f"Test data shape: {model.X_test.shape}")
    """
    # Validate inputs
    X, y = validate_input(X, y)
    
    # Model mapping
    model_classes = {
        "Linear": Linear,
        "Quadratic": Quadratic, 
        "Cubic": Cubic
    }
    
    if model not in model_classes:
        raise ValueError(f"Unknown model type: {model}. Available options: {list(model_classes.keys())}")
    
    # Handle data splitting
    if split_ratio is not None:
        X_train, X_test, y_train, y_test = split_data(X, y, split_ratio, random_state)
        use_split = True
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        use_split = False
    
    # Initialize and fit the model
    model_class = model_classes[model]
    fitted_model = model_class(**kwargs)
    fitted_model.fit(X_train, y_train)
    
    # Store test data as model attributes if split was used
    if use_split:
        fitted_model.X_test = X_test
        fitted_model.y_test = y_test
        fitted_model.X_train = X_train
        fitted_model.y_train = y_train
        fitted_model.split_ratio = split_ratio
        fitted_model.random_state = random_state
    
    # Prepare results
    results = {
        "model_type": model,
        "n_features": X_train.shape[1],
        "n_samples_total": X.shape[0],
        "coefficients": getattr(fitted_model.model, 'coef_', None),
        "intercept": getattr(fitted_model.model, 'intercept_', None)
    }
    
    if use_split:
        # Calculate scores for both train and test sets
        train_r2 = fitted_model.score(X_train, y_train)
        test_r2 = fitted_model.score(X_test, y_test)
        
        results.update({
            "train_r2_score": train_r2,
            "test_r2_score": test_r2,
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "split_ratio": split_ratio,
            "random_state": random_state
        })
        
        # Overall score is test score when split is used
        results["r2_score"] = test_r2
    else:
        # No split, use full dataset score
        results["r2_score"] = fitted_model.score(X_train, y_train)
        results["n_samples_train"] = X_train.shape[0]
    
    return fitted_model, results