from vector import Vector

"""
This class implements the kNN algorithm, with inputs as Vector objects.
"""

class kNN:

    def __init__(self, k=1):
        """
        Initialize the hyperparameters.

        k - number of neighbors to poll
        """
        raise NotImplementedError

    def fit(self, x, y):
        """
        Fit the model to the given data.

        x (Vector) - input vectors to fit
        y (Vector) - output classes to fit
        """
        raise NotImplementedError

    def predict(self, query):
        """
        Predict the class of a given data point or data points.

        query - data point or points to predict the class of
        """
        raise NotImplementedError
