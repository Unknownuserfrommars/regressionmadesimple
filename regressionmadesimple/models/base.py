"""
Enhanced base class for all regression models in RegressionMadeSimple v3.0.0
Provides common functionality: train/test splitting, serialization, scoring, and plotting
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings


class BaseModel(ABC):
    """
    Abstract base class for all regression models.
    Provides common functionality to eliminate code duplication.
    """
    
    def __init__(
        self,
        dataset: Union[pd.Series, pd.DataFrame],
        colX: str,
        colY: str,
        testsize: float = 0.15,
        randomstate: int = 1,
        train_test_split: bool = True,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None
    ):
        """
        Initialize base model with common parameters.
        
        Args:
            dataset: The dataset to input, supports pandas DataFrame
            colX: the X column (input) for regression
            colY: the y column (output) for regression
            testsize: The size of the train_test_split, default 0.15
            randomstate: The random state of train_test_split, default 1
            train_test_split: Whether to split the dataset. Defaults to True
            X_train, y_train, X_test, y_test: Pre-split data (optional)
        """
        self.dataset = dataset
        self.colX = colX
        self.colY = colY
        self.testsize = testsize
        self.randomstate = randomstate
        self.use_split = train_test_split
        
        # Extract X and y from dataset
        self.X = pd.DataFrame(dataset[colX])
        self.y = pd.DataFrame(dataset[colY])
        
        # Handle train/test splitting
        if self.use_split:
            self.X_train, self.X_test, self.y_train, self.y_test = ms.train_test_split(
                self.X, self.y, test_size=self.testsize, random_state=self.randomstate
            )
        elif all(arg is not None for arg in [X_train, y_train, X_test, y_test]):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        else:
            # No splitting - use entire dataset for training
            self.X_train = self.X
            self.y_train = self.y
            self.X_test = None
            self.y_test = None
        
        # Placeholder for model (to be set by subclasses)
        self.model = None
        self.y_pred_tts = None
        self.mse_tts = None
    
    @abstractmethod
    def fit(self):
        """
        Fit the model. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    @abstractmethod
    def predict(self, X_new: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def _compute_metrics(self):
        """
        Compute metrics after fitting (if test data is available).
        Should be called by subclasses after fitting.
        """
        if self.X_test is not None and self.y_test is not None and self.model is not None:
            self.y_pred_tts = self.predict(self.X_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)
    
    def r2_score(self) -> Optional[float]:
        """
        Calculate R² (coefficient of determination) score.
        
        Returns:
            R² score if test data is available, None otherwise
        """
        if self.y_test is not None and self.y_pred_tts is not None:
            return r2_score(self.y_test, self.y_pred_tts)
        return None
    
    def mae(self) -> Optional[float]:
        """
        Calculate Mean Absolute Error.
        
        Returns:
            MAE if test data is available, None otherwise
        """
        if self.y_test is not None and self.y_pred_tts is not None:
            return mean_absolute_error(self.y_test, self.y_pred_tts)
        return None
    
    def rmse(self) -> Optional[float]:
        """
        Calculate Root Mean Squared Error.
        
        Returns:
            RMSE if test data is available, None otherwise
        """
        if self.mse_tts is not None:
            return np.sqrt(self.mse_tts)
        return None
    
    def mse(self) -> Optional[float]:
        """
        Get Mean Squared Error.
        
        Returns:
            MSE if test data is available, None otherwise
        """
        return self.mse_tts
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk using joblib.
        
        Args:
            filepath: Path where the model should be saved
            
        Example:
            >>> model = Linear(data, 'x', 'y')
            >>> model.save_model('my_model.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire model object
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded model instance
            
        Example:
            >>> model = BaseModel.load_model('my_model.pkl')
            >>> predictions = model.predict(new_data)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's performance and parameters.
        Should be overridden by subclasses to include model-specific info.
        
        Returns:
            Dictionary containing model summary
        """
        return {
            'mse': self.mse_tts,
            'rmse': self.rmse(),
            'mae': self.mae(),
            'r2_score': self.r2_score(),
        }
    
    def plot_y_train(self):
        """
        Plot training/test data and predictions.
        Should be implemented by subclasses with model-specific plotting logic.
        """
        raise NotImplementedError("Subclasses should implement plot_y_train()")
    
    def plot_predict(self, X_new: Union[np.ndarray, pd.DataFrame], 
                     y_new: Union[np.ndarray, pd.DataFrame]):
        """
        Plot predictions on new data.
        Should be implemented by subclasses with model-specific plotting logic.
        """
        raise NotImplementedError("Subclasses should implement plot_predict()")