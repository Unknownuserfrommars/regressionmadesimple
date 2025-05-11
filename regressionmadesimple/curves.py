# Still in early dev version. Unstable and not ready for production.
# Few tests have been done.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class CustomCurve:
    def __init__(self, dataset, colX, colY, basis_funcs=None, testsize=0.15, randomstate=1,
                 train_test_split=True, X_train=None, y_train=None, X_test=None, y_test=None):

        self.X = pd.DataFrame(dataset[colX])
        self.y = pd.DataFrame(dataset[colY])

        # Default basis functions (identity + cos + sin)
        if basis_funcs is None:
            basis_funcs = ["x", "cos(x)", "sin(x)"]
        self.basis_funcs = basis_funcs

        self.testsize = testsize
        self.randomstate = randomstate

        # Build design matrix
        self.X_transformed = self._transform_features(self.X.values)

        if train_test_split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_transformed, self.y, test_size=self.testsize, random_state=self.randomstate
            )
        elif all([X_train is not None, y_train is not None, X_test is not None, y_test is not None]):
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
        else:
            self.X_train, self.y_train = self.X_transformed, self.y
            self.X_test = self.y_test = None

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        if self.X_test is not None and self.y_test is not None:
            self.y_pred_tts = self.model.predict(self.X_test)
            self.mse_tts = mean_squared_error(self.y_test, self.y_pred_tts)

    def _transform_features(self, X):
        # X: np.ndarray of shape (n_samples, 1)
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
        return np.hstack(features)

    def predict(self, X_new):
        X_transformed = self._transform_features(np.array(X_new))
        return self.model.predict(X_transformed)

    def summary(self):
        return {
            "basis_functions": self.basis_funcs,
            "coefficients": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "mse": getattr(self, "mse_tts", None)
        }
