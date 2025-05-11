import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# import warnings


class Preworks:
    @staticmethod
    def readcsv(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def create_random_dataset(nrows:int, ncols:int, randrange:tuple, colnames:list):
        df = pd.DataFrame(np.random.randint(randrange[0], randrange[1], size=(nrows, ncols)), columns=colnames)
        return df
    
    @staticmethod
    def split(df, target, test_size=0.2, random_state=42):
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def encode(df: pd.DataFrame):
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders

import logging

class Logger:
    _instance = None

    def __init__(self, enabled=True, level=logging.INFO, name="mymodule"):
        self.enabled = enabled
        self.name = name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

        Logger._instance = self if enabled else NullLogger()

    @classmethod
    def get(cls):
        """Safe access from anywhere; returns NullLogger if not enabled"""
        return cls._instance or NullLogger()

    def debug(self, msg):     self.logger.debug(msg)
    def info(self, msg):      self.logger.info(msg)
    def warning(self, msg):   self.logger.warning(msg)
    def error(self, msg):     self.logger.error(msg)
    def critical(self, msg):  self.logger.critical(msg)


class NullLogger:
    """No-op logger that does nothing; avoids `if logger:` checks everywhere."""
    def debug(self, msg):     pass
    def info(self, msg):      pass
    def warning(self, msg):   pass
    def error(self, msg):     pass
    def critical(self, msg):  pass
