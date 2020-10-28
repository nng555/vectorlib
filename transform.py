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
        self.shape = []
        self.shape = transform.shape
        if(len(self.shape)==2):
            self.transform = transform
            self.in_dim = self.shape[0]
            self.out_dim = self.shape[1]
        else:
            raise Exception('not 2D arrays')

    @classmethod
    def random_init(cls, in_dim, out_dim):
        """
        Initialize the transformation matrix randomly as
        in_dim x out_dim

        in_dim - transformation input dimension
        out_dim - transformation output dimension
        """
        transform = Vector(shape=[in_dim, out_dim])

        return cls(transform)

    def forward(self, v):
        """
        Apply the linear transformation to a vector.
        Verify first that the dimensions are compatible.
        Then apply the transformation as W(v.T)

        v - vector to apply transformation to

        returns - transformed vector
        """

        # transform x vector
        #initilizae vals with rows=rows of transform and columns=columns of v
        return v.twod_matmul(self.transform)


class LinearTransformTest(unittest.TestCase):

    def setUp(self):
        self.v1 = Vector([1, 3, 2])
        self.t1 = LinearTransform(
                transform=[[1, 2,3],
                            [4,5,6]])

        print(self.t1)
        print(self.t1.transform)
        print(self.v1.values)
        self.random = LinearTransform.random_init(4,2)
        print(self.random)

    def testTransform(self):
        tmp = Vector([13, 31])
        print ((self.t1.forward(self.v1)).values)
        self.assertEqual(tmp.values, (self.t1.forward(self.v1)).values)

if __name__ == "__main__":
    unittest.main()
    #print(v1.shape)
