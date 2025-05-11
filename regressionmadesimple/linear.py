from .base_class import BaseModel
from .options import options

import sklearn.model_selection as ms # Avoid naming conflicts with the train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import warnings
from .utils_preworks import Logger

class Linear(BaseModel):
    def __init__(self, dataset: pd.Series|pd.DataFrame, colX:str, colY:str, testsize=0.15, randomstate:int=1, train_test_split:bool=True, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Initializes the Linear Regression model constructor
        Args:
            dataset: The dataset to input, supports pandas DataFrame
            colX: the X column (input) in LinearRegression
            colY: the y column (output) in LinearRegression
            testsize: The size of the train_test_split, default 0.15. Only works if train_test_split is set to True (default)
            randomstate: The random state of train_test_split, default 1. Only works if train_test_split is set to True (default)
            train_test_split: Whether to split the dataset to train. Defaults to true. If the dataset is small, it is recommended to set this to False
            X_train, y_train, X_test, y_test: The provided already split data. Only works if all 4 of these parameters are not None and train_test_split is False
        """
        self.dataset = dataset
        self.colX = colX
        self.colY = colY
        self.testsize = testsize if testsize is not None else options.training.test_size
        self.randomstate = randomstate if randomstate is not None else options.training.random_state
        self.use_split = train_test_split if train_test_split is not None else options.linear.auto_split
        self.X = pd.DataFrame(dataset[colX])
        self.y = pd.DataFrame(dataset[colY])
        if self.use_split:
            self.X_train, self.X_test, self.y_train, self.y_test = ms.train_test_split(self.X, self.y, test_size=self.testsize, random_state=self.randomstate)
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred_tts = self.model.predict(self.X_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)
        elif all(arg is not None for arg in [X_train, y_train, X_test, y_test]):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            self.y_pred_tts = self.model.predict(self.X_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)
        else:
            self.model = LinearRegression()
            self.model.fit(self.X, self.y)

    def plot_y_train(self):
        """
        Note that this only returns the plotly figure. Please use fig.show() yourself.
        This only works if you have the `train_test_split` parameter set to True (default) in the constructor.
        """
        if not hasattr(self, 'y_pred_tts'):
            raise Exception('Please set the `train_test_split` parameter to True in the constructor to use this function.')
        if options.plot.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.X_test, y=self.y_test, mode='markers', name='Test Data'))
            fig.add_trace(go.Scatter(x=self.X_test, y=self.y_pred_tts, mode='lines', name='Prediction'))
            return fig
        elif options.plot.backend == 'matplotlib':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
            ax.plot(self.X_test, self.y_pred_tts, color='red', label='Prediction')
            ax.set_xlabel(self.colX)
            ax.set_ylabel(self.colY)
            ax.legend()
            return fig
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")
    
    def predict(self, X_new: np.ndarray|pd.DataFrame):
        return self.model.predict(X_new)
    
    def plot_predict(self, X_new: np.ndarray|pd.DataFrame, y_new: np.ndarray|pd.DataFrame):
        """
        Note that this only returns the plotly figure. Please use fig.show() yourself.
        """
        if options.plot.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_new, y=y_new, mode='markers', name='New Data'))
            fig.add_trace(go.Scatter(x=X_new, y=self.predict(X_new), mode='lines', name='Prediction'))
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")
        return fig
    
    def mse(self):
        warnings.warn(
        "This function is deprecated and will be removed in a future version. Use summary() instead.",
        DeprecationWarning,
        stacklevel=2
        )

        if not self.mse:
            raise Exception('Please set the `train_test_split` parameter to True in the constructor to use this function.')
        return self.mse_tts
    
    def summary(self):
        return {
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'mse': getattr(self, 'mse_tts', None)
        }