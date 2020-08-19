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

## Testing your code

To test your code, I have provided some unit tests for each of the basic functionalities. You may want to add your own, or use them to test the problems in the Linear Algebra textbook. To run the tests, simply run the corresponding file, like `python vector.py`.

# kNN

Now that we are familiar with vectors and linear transformations and have a functioning library, we will implement our first algorithm, k-Nearest Neighbors!

This will be implemented in the file `knn.py`, which includes a class `kNN` with three methods you will have to implement.

1. `__init__` will construct the raw model including any hyperparameters we need to set (specifically k)

2. `fit` will take in a training dataset (x, y) which should be a 2-d and 1-d vector containing the input vectors and output labels, and should store them for prediction later.

3. `predict` will take in a query point (1-d vector) or query points (2-d vector) and return predictions for all points using the kNN algorithm.

## UCI Iris Dataset

To test our algorithm, we will use the [UCI Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/). This dataset contains 150 data points of flower measurements and flower names. Our task will be to learn to predict the flower name based on its measurements.
After running the experiments, you will write up your results in a report, just like a real scientist! I will provide a template later.

### Preprocessing the Data

The first step of any experiments we run will be to download and preprocess your data. 

1. Download the file `iris.data` from the link above. We will store this in a new `data` folder.

2. Complete the script `preprocess.py` which will take in an input file of raw data and output our preprocessed training, validation, and test data.

### Tuning the Hyperparameters

Once our data is preprocessed, we can start running experiments on our data.

## UCI Wine Dataset

To see how the scale of different parameters affects our kNN algorithm, we will use the [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine). This dataset contains 178 data points of wine measurements and labels of where they came from. Again, our task will be to predict which area each wine came from. However, this problem is a little harder because we have 13 features instead of 4, and each will have different scales. We will perform kNN with and without feature normalization
and analyze how our performance changes.

### Preprocessing the Data

Preprocess the data like you did for the Iris dataset. Since all we need to change is how each row is changed into features and labels, fill in the `process_wine_row` method. Once you have this written, you can run `preprocess.py` on the wine data and get the training, validation, and test vectors!

### kNN without Normalization

First try running kNN without normalizing any of the data and use the raw feature values. Tune your `k` value on the validation set then test it on the test set and write down the test accuracy. Does our classifier seem to be able to do well?

### kNN with Normalization

Now we will normalize our features. In the Vector class complete the `normalize` function. This function operates in place, meaning that we will change the vector itself rather than returning a new Vector. Once we have this function complete, use the `normalize` parameter in `preprocess.py` to add logic to normalize the vectors before we write them to disk.

Finally, run our experiments on Iris and Wine again, but this time normalizing our features. How do the results change with and without normalization?
