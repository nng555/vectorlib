import random
import unittest
import numpy as np

from vector import Vector

"""
This class implements a linear transformation operation that
uses a 2d python array as its underlying data structure.
"""

class LinearTransform:

    def get_shape(self,values):
        '''
        Runs through the values and counts the number of elements
        per dimension
        Returns final shape list when it's no longer a list (past 1D)
        '''
        cntr = 0
        if not type(values) == list:
            return self.shape
        else:
            for i in range(len(values)):
                cntr += 1
            self.shape.append(cntr)
            return self.get_shape(values[0])

    def __init__(self, transform):
        """
        Initialize the transformation matrix

        transform - the matrix used for this transformation
        """
        self.shape = []
        self.shape = self.get_shape(transform)

        if(len(self.shape)==2):
            self.transform = transform
            self.in_dim = len(self.transform)
            self.out_dim = len(self.transform[0])
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
        transform = [[(0) for i in range(out_dim)]for x in range(in_dim)]

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
        vals = [[(0) for i in range(v.shape[1])]for x in range(len(self.transform))]

        if (self.out_dim == v.shape[0]):

            if(len(v.shape)==1):
                self.shape.insert(0,1)

            #rows of first matrix
            for i in range(len(self.transform)):
                #columns of second matrix
                for j in range(v.shape[1]):
                    #rows of second matrix
                    for k in range(v.shape[0]):
                        vals[i][j] += self.transform[i][k] * v.values[k][j]

            transV = Vector(vals,None)
            return transV

        else:
            raise Exception('not matching dimensions')

        '''
        # vector x transform
        #initilizae vals with rows=rows of v and columns=columns of transform

        if(len(v.shape)==1):
            v.shape.insert(0,1)

        vals = [[(0) for i in range(self.out_dim)]for x in range(v.shape[0])]

        if (v.shape[1] == self.in_dim):

            #rows of first matrix
            for i in range(v.shape[0]):
                #columns of second matrix
                for j in range(self.out_dim):
                    #rows of second matrix
                    for k in range(self.in_dim):
                        vals[i][j] += v.values[i][k] * self.transform[k][j]

            transV = Vector(vals,None)
            return transV

        else:
            raise Exception('not matching dimensions')'''

class LinearTransformTest(unittest.TestCase):

    def setUp(self):
        self.v1 = Vector([1, 3, 2])
        self.t1 = LinearTransform(
                transform=[[1, 2],
                            [3,4],
                            [5,6]])
        self.v2 = Vector([[1], [3], [2]])
        self.t2 = LinearTransform(
                transform=[[1, 2,3],
                            [4,5,6]])

        print(self.t1)
        print(self.t1.transform)
        print(self.v1.values)
        self.random = LinearTransform.random_init(4,2)
        print(self.random)

    def testTransform(self):
        tmp = Vector([[13], [31]])
        print ((self.t2.forward(self.v2)).values)

        #print ((self.t1.forward(self.v1)).values)
        self.assertEqual(tmp.values, (self.t2.forward(self.v2)).values)

if __name__ == "__main__":
    unittest.main()
    #print(v1.shape)
