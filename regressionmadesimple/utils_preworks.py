import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import warnings

class preworks:
    @staticmethod
    def readcsv(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def _create_random_dataset(nrows:int, ncols:int, randrange:tuple, colnames:list):
        warnings.warn(
        "This function is deprecated and will be removed in a future version. Use crd() instead.",
        DeprecationWarning,
        stacklevel=2
        )
        df = pd.DataFrame(np.random.randint(randrange[0], randrange[1], size=(nrows, ncols)), columns=colnames)
        return df
    
    @staticmethod
    def crd(**kwargs):
        return preworks._create_random_dataset(**kwargs)
    
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