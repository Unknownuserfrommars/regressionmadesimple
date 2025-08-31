"""
Input/Output utilities for RegressionMadeSimple.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union, Optional


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save a fitted model to disk.
    
    Parameters
    ----------
    model : object
        Fitted model instance to save
    filepath : str or Path
        Path where to save the model
        
    Raises
    ------
    ValueError
        If model is not fitted
    """
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        raise ValueError("Model must be fitted before saving")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load a model from disk.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file
        
    Returns
    -------
    model : object
        Loaded model instance
        
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def export_results(results: Dict, filepath: Union[str, Path], format: str = 'json') -> None:
    """
    Export model results to file.
    
    Parameters
    ----------
    results : dict
        Results dictionary to export
    filepath : str or Path
        Output file path
    format : str, default='json'
        Export format ('json' or 'csv')
        
    Raises
    ------
    ValueError
        If format is not supported
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
    elif format.lower() == 'csv':
        # Convert to DataFrame and save as CSV
        df_data = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    df_data[key] = value
                else:
                    # Flatten multi-dimensional arrays
                    df_data[key] = value.flatten()
            else:
                df_data[key] = [value]  # Make scalar values into lists
        
        # Ensure all columns have same length
        max_len = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 
                     for v in df_data.values())
        
        for key, value in df_data.items():
            if isinstance(value, list) and len(value) == 1:
                df_data[key] = value * max_len
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")


def load_data(filepath: Union[str, Path], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    filepath : str or Path
        Path to data file
    **kwargs : dict
        Additional arguments passed to pandas read functions
        
    Returns
    -------
    data : np.ndarray or pd.DataFrame
        Loaded data
        
    Raises
    ------
    ValueError
        If file format is not supported
    FileNotFoundError
        If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif suffix == '.json':
        return pd.read_json(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    elif suffix == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif suffix in ['.npy']:
        return np.load(filepath)
    elif suffix == '.npz':
        return np.load(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_predictions(
    predictions: np.ndarray, 
    filepath: Union[str, Path], 
    X: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None
) -> None:
    """
    Save model predictions to file.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    filepath : str or Path
        Output file path
    X : np.ndarray, optional
        Input features used for predictions
    feature_names : list, optional
        Names of input features
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {'predictions': predictions}
    
    if X is not None:
        if feature_names is not None:
            for i, name in enumerate(feature_names):
                data[name] = X[:, i]
        else:
            for i in range(X.shape[1]):
                data[f'feature_{i}'] = X[:, i]
    
    df = pd.DataFrame(data)
    
    if filepath.suffix.lower() == '.csv':
        df.to_csv(filepath, index=False)
    elif filepath.suffix.lower() == '.json':
        df.to_json(filepath, orient='records', indent=2)
    else:
        # Default to CSV
        df.to_csv(filepath.with_suffix('.csv'), index=False)