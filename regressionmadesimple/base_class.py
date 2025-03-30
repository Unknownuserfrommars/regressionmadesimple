class BaseModel:
    def fit(self):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError