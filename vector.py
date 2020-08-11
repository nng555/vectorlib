import random
import unittest
import copy
import numpy as np

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

    def l2(self, v2):
        """
        Calculate the l2 distance to the vector v2

        v2 - vector to calculate distance to

        returns - distance to v2
        """

        raise NotImplementedError

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

        else:
            switch = self.matrix([self.shape[1], self.shape[0]])
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    switch[j][i] = self.values[i][j]

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
        self.assertAlmostEqual(self.v1.l2(self.v2), 4.0)
        self.assertAlmostEqual(self.v2.l2(self.v3), 3.4641)

if __name__ == "__main__":
    unittest.main(verbosity=2)

    # TODO: If you want to print test values, you can do it inside the tests...
    '''test = Vector([[1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6],
          [4, 5, 6, 7]],None)
    print(test.values)
    temp = test.copy_init(test)
    print(temp.values)

    tmp = Vector([[1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6],
          [4, 5, 6, 7]], None)
    print (tmp.shape)

    # TODO: the __eq__ method overrides the `==` operator, so you can
    # call it like `test == tmp`
    print(test.__eq__(tmp))'''

    #print (test.copy_init(test))
