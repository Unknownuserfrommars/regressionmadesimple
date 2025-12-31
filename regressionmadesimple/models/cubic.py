"""
Cubic Regression Model - Refactored for v3.0.0
Uses enhanced BaseModel and PolynomialFeatures
"""

from typing import Union, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go

from .base import BaseModel
from ..options import options


class Cubic(BaseModel):
    """
    Cubic (Polynomial degree=3) Regression model.
    Inherits common methods from BaseModel.
    """
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        colX: str,
        colY: str,
        testsize: Optional[float] = None,
        randomstate: Optional[int] = None,
        train_test_split: Optional[bool] = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Cubic Regression model.
        
        Args:
            dataset: The dataset to input, supports pandas DataFrame
            colX: the X column (input)
            colY: the y column (output)
            testsize: The size of the train_test_split, default from options or 0.15
            randomstate: The random state of train_test_split, default from options or 1
            train_test_split: Whether to split the dataset. Defaults to True
            X_train, y_train, X_test, y_test: Pre-split data (optional)
        """
        # Use options if parameters not provided
        testsize = testsize if testsize is not None else options.training.test_size
        randomstate = randomstate if randomstate is not None else options.training.random_state
        train_test_split = train_test_split if train_test_split is not None else options.quadratic.auto_split
        
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
        
        # Polynomial feature transformer
        self.poly_features = PolynomialFeatures(degree=3)
        
        # Fit the model
        self.fit()
    
    def fit(self):
        """
        Fit the cubic regression model.
        """
        # Transform features
        X_poly_train = self.poly_features.fit_transform(self.X_train)
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_poly_train, self.y_train)
        
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
        X_poly_new = self.poly_features.transform(X_new)
        return self.model.predict(X_poly_new)
    
    def plot_y_train(self):
        """
        Plot training/test data and predictions.
        
        Returns:
            Plotly or Matplotlib figure object
        """
        if not hasattr(self, 'y_pred_tts') or self.y_pred_tts is None:
            raise Exception('Please set the `train_test_split` parameter to True in the constructor to use this function.')
        
        if options.plot.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.X_train[self.colX].values.flatten(),
                y=self.y_train[self.colY].values.flatten(),
                mode='markers',
                name='Train Data'
            ))
            fig.add_trace(go.Scatter(
                x=self.X_test[self.colX].values.flatten(),
                y=self.y_test[self.colY].values.flatten(),
                mode='markers',
                name='Test Data'
            ))
            fig.add_trace(go.Scatter(
                x=self.X_test[self.colX].values.flatten(),
                y=self.y_pred_tts.flatten(),
                mode='lines',
                name='Predicted Data'
            ))
            fig.update_layout(
                title='Cubic Regression Model - Train and Test Data',
                xaxis_title=self.colX,
                yaxis_title=self.colY
            )
            return fig
        elif options.plot.backend == 'matplotlib':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(self.X_train[self.colX], self.y_train[self.colY], color='blue', label='Train Data')
            ax.scatter(self.X_test[self.colX], self.y_test[self.colY], color='red', label='Test Data')
            ax.plot(self.X_test[self.colX], self.y_pred_tts, color='green', label='Predicted Data')
            ax.set_title('Cubic Regression Model - Train and Test Data')
            ax.set_xlabel(self.colX)
            ax.set_ylabel(self.colY)
            ax.legend()
            return fig
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")
    
    def plot_predict(self, X_new: Union[np.ndarray, pd.DataFrame], 
                     y_new: Union[np.ndarray, pd.DataFrame]):
        """
        Plot predictions on new data.
        
        Args:
            X_new: New input data
            y_new: Actual output for new data
            
        Returns:
            Plotly figure object
        """
        if options.plot.backend == 'plotly':
            predictions = self.predict(X_new)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_new[self.colX] if isinstance(X_new, pd.DataFrame) else X_new.flatten(),
                y=y_new[self.colY] if isinstance(y_new, pd.DataFrame) else y_new.flatten(),
                mode='markers',
                name='New Data'
            ))
            fig.add_trace(go.Scatter(
                x=X_new[self.colX] if isinstance(X_new, pd.DataFrame) else X_new.flatten(),
                y=predictions.flatten(),
                mode='lines',
                name='Prediction'
            ))
            fig.update_layout(
                title='Cubic Regression Model - New Data Prediction',
                xaxis_title=self.colX,
                yaxis_title=self.colY
            )
            return fig
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")
    
    def summary(self):
        """
        Get model summary including coefficients and metrics.
        
        Returns:
            Dictionary with model information
        """
        base_summary = super().summary()
        base_summary.update({
            'model': 'Cubic Regression',
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
        })
        return base_summary