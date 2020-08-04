import random
import unittest

"""
This class implements a simple vector object that uses a multidimensional
python array as its underlying data structure.

test
"""

class Vector:

    def __init__(self, values=None, shape=None):
        """
        Initialize the vector

        values - the array values to store in the vector
                 if None, then fill randomly based on shape
        shape - an array containing vector dimensions
                if None, then set based on values
        """
        # initialize self.values
        # initialize self.shape

        raise NotImplementedError

    @classmethod
    def copy_init(self, v2):
        """
        Initialize by performing a deep copy of v2

        v2 - vector to copy
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def transpose(self):
        """
        Transpose the vector. If vector is one dimensional
        with shape [x] (which is implicitly [1, x]),
        then its transpose will have shape [x, 1]

        returns - transpose vector
        """
        raise NotImplementedError

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
        tmp = Vector(shape=[6,2])
        self.assertEqual(tmp.shape, [6,2])

    # test the deep copy
    def testCopy(self):
        tmp = Vector(self.v4)
        self.assertEqual(tmp, self.v4)

    # test transposing row to column vector
    def testTranspose(self):
        tmp = Vector([[2], [6], [3], [5]])
        self.assertEqual(tmp, self.v3.transpose())

    # test calculating l2 distance
    def testl2(self):
        self.assertAlmostEqual(self.v1.l2(self.v2), 4.0)
        self.assertAlmostEqual(self.v2.l2(self.v3), 3.4641)

if __name__ == "__main__":
    unittest.main()
