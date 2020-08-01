# vectorlib

To get familiar with vectors, matrices, and transformations, as well as build up a library for use in implementing your own ML algorithms, we will code up an implementation of vectors and linear transformations. We can then use these classes and their functions to write algorithms for nearest neighbour and regression algorithms. The class stubs for vector.py and transform.py have been provided for you.

## vector.py

`vector.py` implements a simple vector with a shape and a set of values. The underlying data structure used to store the values should be a multidimensional python array. The class should have two variables, `values` and `shape`. `values` should be a python array that stores the values of the vector. `shape` stores the dimensions of the vector as a list. For example, a 4 x 4 dimensional matrix will have a shape of [4, 4] and a row vector with 5 values will have shape [5,].

1. First implement the constructor. There are two keyword params, which are optional. If both are set, make sure that the shape and values match. If only values are set, then you will have to set the shape manually. If only the shape is set, then initialize the values randomly.

2. Next implement the copy constructor, which takes an input vector and makes a deep copy

3. Next implement the equals operator, which checks if the shape and values are the same.

4. Next implement the `l2` function which returns the l2 distance between two vectors.

5. Finally implement the `transpose` function, which returns a transposed version of the vector.

## transform.py

`transform.py` implements a simple linear transformation with an underlying 2d python array to store the transformation values. We can initialize it two ways, by specifying a specific transform, or by specifying the input and output dimensions. In all cases the transformation matrix must be two dimensional. If only the dimensions are specified, randomly initialize the values.

1. Implement the constructor. Verify the input matrix is two dimensional.

2. Implement the random init constructor. The resulting transformation matrix should be of the size in_dim x out_dim.

3. Implement the `forward` function. This will pass a vector through the transformation, and return the resulting transformed vector.
