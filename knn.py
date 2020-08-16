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

        # TODO: set self.k

        raise NotImplementedError

    def fit(self, x, y):
        """
        Fit the model to the given data.

        x (Vector) - input vectors to fit
        y (Vector) - output classes to fit
        """

        # TODO: store x and y and make sure they are in the right format
        # x should be 2d with dimensions (# examples x # features)
        # y should be 1d with dimesnsions # examples

        raise NotImplementedError

    def predict(self, query):
        """
        Predict the class of a given data point or data points.

        query - data point or points to predict the class of
        """

        # TODO: loop through every point
        #    TODO: inside loop, calculate the l2 distance between the query
        #          and every input point
        # TODO: sort the distances (np.argsort is useful here)
        #       then return the majority vote label
        raise NotImplementedError
