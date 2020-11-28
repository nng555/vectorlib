import numpy as np

from vector import Vector
from transform import LinearTransform

"""
This class implemenets a regression model as a
"""

class Regression:

    def __init__(self, in_dim, out_dim, lr):
        """
        Initialize the transformation and hyperparameters.

        in_dim - input dimension of model
        out_dim - output dimension of model
        lr - learning rate
        logistic - whether or not to do logistic regression
        """
        self.weights = LinearTransform.random_init(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr

    def forward(self, x):
        # apply the forward pass of our regressor
        out = self.weights.forward(x)
        return out

    def get_loss(self, y, out):

        # calculate loss
        error = out - y
        loss = 0
        numel = 0
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                loss += error[i][j] ** 2
                numel += 1
        loss = loss/numel

        # TODO: calculate gradient
        dLdo = error

        return loss, dLdo

    def backward(self, x, dLdo):
        x.transpose()
        dLdw = x.twod_matmul(dLdo)
        dLdw = dLdw * (1./x.shape[0])

        # TODO: update weights
        grad = dLdw * self.lr
        self.weights.transform = self.weights.transform - grad
