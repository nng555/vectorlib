import random
import unittest
import copy
import numpy as np
from math import sqrt

"""
This class implements a simple vector object that uses a multidimensional
python array as its underlying data structure.

test
"""

class Vector:


    def matrix(self, shape):
        if(len(shape)==1):
            return [(np.random.normal()) for i in range(shape[0])]
        else:
            return [self.matrix(shape[1:])for x in range(shape[0])]

    def __init__(self, values=None, shape=None):
        '''
        Initialize the vector
        values - the array values to store in the vector
                 if None, then fill randomly based on shape
        shape - an array containing vector dimensions
                if None, then set based on values
        '''

        #if shape defined and values are empty
        if(values==None and shape != None):
            self.shape = shape
            self.values = self.matrix(self.shape)

        #if only shape is empty
        elif(values != None and shape == None):
            self.values = values
            # TODO: Not allowed to use np here...you need to get the shape yourself
            self.shape = list(np.shape(self.values))

        # both are defined
        # initialize self.values
        # initialize self.shape
        elif(values != None and shape != None):
            self.values = values
            self.shape = shape
        else:
            raise Exception('Both cannot be none')

        #raise NotImplementedError

    @classmethod
    def copy_init(self, v2):
        """
        Initialize by performing a deep copy of v2

        v2 - vector to copy
        """
        return copy.deepcopy(Vector(v2.values,None))
        #raise NotImplementedError

    def distance(self,vals,shapes):
        """
        recursive method to calculate distance

        returns sum of all of the distances

        """
        num = 0
        sum = 0
        if(len(shapes)==1):
            for i in range(shapes[0]):
                sum += ((self.values[i]-vals[i])**2)
            return sum
        else:
            return num+distance(vals, shape[1:])

    def l2(self, v2):
        """
        Calculate the l2 distance to the vector v2

        v2 - vector to calculate distance to

        returns - distance to v2
        """
        # uses recursive distance method and takes the square root of it
        return sqrt(self.distance(v2.values, v2.shape))

        #raise NotImplementedError

    def __eq__(self, v2):
        """
        Check if the two vectors have the same shape
        and values.

        v2 - vector to check equality with

        returns - whether vectors match or not
        """
        if (self.values == v2.values and self.shape == v2.shape):
            return True
        else:
            return False

        #raise NotImplementedError

    def transpose(self):
        """
        Transpose the vector. If vector is one dimensional
        with shape [x] (which is implicitly [1, x]),
        then its transpose will have shape [x, 1]

        returns - transpose vector
        """
        #just for 1d or 2d arrays
        if(len(self.shape)==1):
            switch = self.matrix([self.shape[0],1])
            for i in range(self.shape[0]):
                switch[i][0] = self.values[i]

        elif (len(self.shape)==2):
            switch = self.matrix([self.shape[1], self.shape[0]])
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    switch[j][i] = self.values[i][j]
        else:
            raise Exception('matrix must be 1d or 2d')

        flip = Vector(switch,None)
        return flip
        #raise NotImplementedError

class VectorTest(unittest.TestCase):
    # vector class unit tests

    # initialize some vectors
    def setUp(self):
        v1 =  [5, 7, 4, 2]
        v2 =  [3, 5, 6, 4]
        v3 =  [2, 6, 3, 5]
        v4 = [[1, 2, 3, 4],
              [2, 3, 4, 5],
              [3, 4, 5, 6],
              [4, 5, 6, 7]]

        self.v1 = Vector(v1)
        self.v2 = Vector(v2)
        self.v3 = Vector(v3)
        self.v4 = Vector(v4)

    # check some shapes
    def testShape(self):
        self.assertEqual(self.v1.shape, [4])
        self.assertEqual(self.v3.shape, [4])
        self.assertEqual(self.v4.shape, [4, 4])
        tmp = Vector(shape=[2,2,2])
        print(tmp.shape)
        print(tmp.values)
        self.assertEqual(tmp.shape, [2,2,2])

    # test the deep copy
    def testCopy(self):
        tmp = Vector.copy_init(self.v4)
        self.assertEqual(tmp, self.v4)

    # test transposing row to column vector
    def testTranspose(self):
        tmp = Vector([[2], [6], [3], [5]])
        print(self.v3.shape)
        self.assertEqual(tmp, self.v3.transpose())

    # test calculating l2 distance
    def testl2(self):
        print(self.v1.l2(self.v2))
        print(self.v2.l2(self.v3))
        self.assertAlmostEqual(self.v1.l2(self.v2), 4.0, places=4)
        self.assertAlmostEqual(self.v2.l2(self.v3), 3.4641, places=4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
