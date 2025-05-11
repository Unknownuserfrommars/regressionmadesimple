from .base_class import BaseModel
from .options import options

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class Quadratic(BaseModel):
    def __init__(self, dataset: pd.DataFrame, colX, colY, testsize=0.15, randomstate=1, train_test_split=True, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Initializes the Quadratic Regression model constructor
        Args:
            dataset: The dataset to input, supports pandas DataFrame
            colX: the X column (input) in Quadratic Regression
            colY: the y column (output) in Quadratic Regression
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
        self.use_split = train_test_split if train_test_split is not None else options.quadratic.auto_split

        self.X = pd.DataFrame(dataset[colX])
        self.y = pd.DataFrame(dataset[colY])

        if self.use_split:
            from sklearn.model_selection import train_test_split as tts
            self.X_train, self.X_test, self.y_train, self.y_test = tts(self.X, self.y, test_size=self.testsize, random_state=self.randomstate)
            poly_features = PolynomialFeatures(degree=2)
            X_poly_train = poly_features.fit_transform(self.X_train)
            X_poly_test = poly_features.transform(self.X_test)

            self.model = LinearRegression()
            self.model.fit(X_poly_train, self.y_train)
            self.y_pred_tts = self.model.predict(X_poly_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)

        elif all(arg is not None for arg in [X_train, y_train, X_test, y_test]):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            poly_features = PolynomialFeatures(degree=2)
            X_poly_train = poly_features.fit_transform(self.X_train)
            X_poly_test = poly_features.transform(self.X_test)

            self.model = LinearRegression()
            self.model.fit(X_poly_train, self.y_train)
            self.y_pred_tts = self.model.predict(X_poly_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)
        else:
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(self.X)
            self.model = LinearRegression()
            self.model.fit(X_poly, self.y)
            self.y_pred_tts = None
            self.mse_tts = None

    def plot_y_train(self):
        """
        Note that this only returns the plotly figure. Please use fig.show() yourself.
        This only works if you have the `train_test_split` parameter set to True in the constructor to use this function.
        """
        if not hasattr(self, 'y_pred_tts'):
            raise Exception('Please set the `train_test_split` parameter to True in the constructor to use this function.')
        if options.plot.backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.X_train[self.colX], y=self.y_train[self.colY], mode='markers', name='Train Data'))
            fig.add_trace(go.Scatter(x=self.X_test[self.colX], y=self.y_test[self.colY], mode='markers', name='Test Data'))
            fig.add_trace(go.Scatter(x=self.X_test[self.colX], y=self.y_pred_tts, mode='lines', name='Predicted Data'))
            fig.update_layout(title=f'Quadratic Regression Model - Train and Test Data', xaxis_title=self.colX, yaxis_title=self.colY)
            return fig
        elif options.plot.backend == 'matplotlib':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(self.X_train, self.y_train, color='blue', label='Train Data')
            ax.scatter(self.X_test, self.y_test, color='green', label='Test Data')
            ax.plot(self.X_test, self.y_pred_tts, color='red', label='Prediction')
            ax.set_xlabel(self.colX)
            ax.set_ylabel(self.colY)
            ax.legend()
            return fig
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")
    
    def predict(self, X_new: np.ndarray|pd.DataFrame):
        poly_features = PolynomialFeatures(degree=2)
        X_poly_new = poly_features.fit_transform(X_new)
        return self.model.predict(X_poly_new)
    
    def plot_predict(self, X_new: np.ndarray|pd.DataFrame, y_new: np.ndarray|pd.DataFrame):
        """
        Note that this only returns the plotly figure. Please use fig.show() yourself.
        """
        if options.plot.backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_new[self.colX], y=y_new[self.colY], mode='markers', name='New Data'))
            fig.add_trace(go.Scatter(x=X_new[self.colX], y=self.predict(X_new), mode='lines', name='Prediction'))
            fig.update_layout(title=f'Quadratic Regression Model - New Data Prediction', xaxis_title=self.colX, yaxis_title=self.colY)
            return fig
        else:
            raise NotImplementedError(f"Plotting backend {options.plot.backend} is not implemented yet.")

    def summary(self):
        return {
            'model': 'Quadratic Regression',
            'mse': self.mse_tts,
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist()
        } if hasattr(self, 'mse_tts') else None    
