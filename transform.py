import random
import unittest
import numpy as np

from vector import Vector

"""
This class implements a linear transformation operation that
uses a 2d python array as its underlying data structure.
"""

class LinearTransform:

    def __init__(self, transform):
        """
        Initialize the transformation matrix

        transform - the matrix used for this transformation
        """
        self.transform = []
        if(len(transform)==2):
            self.transform = transform
        else:
            raise Exception('not 2D arrays')

    @classmethod
    def random_init(cls, in_dim, out_dim):
        """
        Initialize the transformation matrix randomly as
        out_dim x in_dim

        in_dim - transformation input dimension
        out_dim - transformation output dimension
        """
        transform = []
        for i in range(in_dim):
            for j in range(out_dim):
                transform[i][j] = np.random.normal()

        return cls(transform)

    def forward(self, v):
        """
        Apply the linear transformation to a vector.
        Verify first that the dimensions are compatible.
        Then apply the transformation as W(v.T)

        v - vector to apply transformation to

        returns - transformed vector
        """

class LinearTransformTest(unittest.TestCase):

    def setUp(self):
        self.v1 = Vector([1, 3, 2])
        self.t1 = LinearTransform(
                transform=[[1, 2, 3],
                           [4, 5, 6]])
        print(self.t1)
        self.random = LinearTransform.random_init(4,2)
        print(self.random)

    def testTransform(self):
        tmp = Vector([13, 31])
        self.assertEqual(tmp, self.t1.forward(self.v1))


if __name__ == "__main__":
    unittest.main()
