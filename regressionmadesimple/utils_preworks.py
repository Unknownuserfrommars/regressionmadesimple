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