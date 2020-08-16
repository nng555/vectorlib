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

    def matrix(self, shape):
        '''
        Goes through the shape list until it reaches the last dimension
        and recurisvly builds lists within lists to produce the values
        Values are random
        '''
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
            self.shape = []
            self.shape = self.get_shape(self.values)


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
    def copy_init(cls, v2):
        """
        Initialize by performing a deep copy of v2

        v2 - vector to copy
        """
        return cls(copy.deepcopy(v2.values))
        #raise NotImplementedError

    def distance(self,v2,temp):
        """
        recursive method to calculate distance

        returns sum of all of the distances

        """
        sum = 0
        if (len(v2.shape) == 1):
            for i in range(v2.shape[0]):
                sum += ((temp.values[i] - v2.values[i])**2)
            return sum
        else:
            for i in range(v2.shape[0]):
                tmp1 = Vector(v2.values[i], v2.shape[1:])
                tmp2 = Vector(temp.values[i], temp.shape[1:])
                sum += self.distance(tmp1,tmp2)
            return sum

    def l2(self, v2):
        """
        Calculate the l2 distance to the vector v2

        v2 - vector to calculate distance to

        returns - distance to v2
        """
        # uses recursive distance method and takes the square root of it
        temp = self.copy_init(self)
        return self.distance(v2,temp)

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

            if(self.shape[1] == 1):
                switch = self.matrix([self.shape[0]])
                for i in range(self.shape[0]):
                        switch[i] = self.values[i][0]
            else:
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
        v5 = [[[2,10],[2,10]],[[3,3],[3,3]]]
        v6 = [[[4,4],[2,2]],[[5,1],[6,3]]]


        self.v1 = Vector(v1)
        self.v2 = Vector(v2)
        self.v3 = Vector(v3)
        self.v4 = Vector(v4)
        self.v5 = Vector(v5)
        self.v6 = Vector(v6)

    # check some shapes
    def testShape(self):
        self.assertEqual(self.v1.shape, [4])
        self.assertEqual(self.v3.shape, [4])
        self.assertEqual(self.v4.shape, [4, 4])
        tmp = Vector(shape=[2,2,2])
        self.assertEqual(tmp.shape, [2,2,2])

    # test the deep copy
    def testCopy(self):
        tmp = Vector.copy_init(self.v4)
        self.assertEqual(tmp, self.v4)

    # test transposing row to column vector
    def testTranspose(self):
        tmp = Vector([[2], [6], [3], [5]])
        self.assertEqual(tmp, self.v3.transpose())

    # test calculating l2 distance
    def testl2(self):
        self.assertAlmostEqual(self.v1.l2(self.v2), 16.0, places=4)
        self.assertAlmostEqual(self.v2.l2(self.v3), 12.0, places=4)
        print(self.v5.l2(self.v6))

if __name__ == "__main__":
    unittest.main(verbosity=2)
