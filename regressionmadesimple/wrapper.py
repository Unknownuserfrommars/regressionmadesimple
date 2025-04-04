from .linear import Linear
from .quadratic import Quadratic
from .cubic import Cubic

class LinearRegressionModel:
    """
    Wrapper for fitting different regression types: 'linear', 'quadratic', 'cubic'
    """

    @staticmethod
    def fit(dataset, colX, colY, model='linear', **kwargs):
        """
        Parameters:
            - dataset: pd.DataFrame
            - colX: str
            - colY: str
            - model: 'linear' | 'quadratic' | 'cubic'
            - kwargs: extra args passed to model class (like testsize, randomstate)
        """
        model_name = model.strip().lower().capitalize()
        model_class = globals().get(model_name)

        if not model_class:
            raise ValueError(f"Model '{model}' not supported.")
        return model_class(dataset, colX, colY, **kwargs)