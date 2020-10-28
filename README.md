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

# Linear Regression using Backpropagation

For our final model we will build a simple linear regressor that learns via backpropagation. Our model will be implemented in `regression.py`. `forward` will calculate our predicted values for each input. `get_loss` will return the loss of our model and the gradient of our loss with respect to our model output. This gradient is used in `backward` in order to move our weights towards lower loss.

## Backpropagation

1. First complete the gradient calculation of dL/do. Remember that our loss metric is mean squared error.

2. Then complete the gradient update of the weights using dL/dw. Since our gradient points us in the positive direction, we will need move our weights in the opposite direction of the gradient.

## Seoul Bike Sharing

We will be predicting bike share demand using the [Seoul Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand). This dataset contains 8760 data points of various weather metrics in Seoul during particular times of day, and we would like to predict how many bikes we expect to be used. We can throw away the date feature since it will be hard to encode. We will use the rest of the features. Look through the website and `bike.data` to see how the values
correspond.

1. Season information is contained in `row[-3]`. Instead of representing the seasons with numbers, we will use a one-hot vector that has 4 elements. A `1` in one of these elements will represent each season. For example, `[0,0,0,1]` can represent Winter, and `[0,1,0,0]` can represent Summer. Fill out the extension of the feature data with the season features.

2. Holiday information is contained in `row[-2]`. Add a feature that is a 0 if there is no holiday, and a 1 if there is a holiday.

3. Bike functionality information is contained in `row[-1]`. Add a feature that is 0 if the bikes are not working, and a 1 if they are working.

4. To simplify our model, we include the model bias as a feature. Add a bias feature that is 1 for every example.

Finally, preprocess our data into a data directory, making sure to pass in the normalization flag.

## Training Loop

Now that our data is preprocessed we will complete the training loop. This loop will go through our data multiple times and update our model. Every time our model "sees" all of our data once, that is called one epoch. Since we don't want our model to see all the data at once, we randomly select small chunks of data at a time, called batches. Notice the two loops on lines 46 and 55 that loop over our epochs and batches.

1. Initialize our model. Using `x_train` and `y_train`, find out the number of features and number of outputs then use these to initialize our regression model

2. Once we have our `x_batch`  we need to pass it through our model. Fill out a single line of code to generate predictions on our inputs.

3. With our inputs, we can calculate how bad our model is, and how we can improve it. Fill out a single line of code to generate our model loss and gradient.

4. With our gradient, we can update our model to make it better. Fill out a single line of code to run backpropagation with our gradients.

5. Every epoch, we want to monitor how well our model is doing. Make predictions and calculate the loss on the validation set.

## Analyzing the Model

With our training loop complete, we can learn a model with backpropagation! Start with training for 10 epochs and a batch size of 128. Try the following learning rates: `[1, 0.1, 0.001, 0.0001, 0.00001]`. 

1. What happens to our validation loss when our learning rate is too high? What about when our learning rate is too low?

2. Look at the weights of the model after it is done training with the best performing learning rate. What does a positive weight mean? What does a negative weight mean? Are these numbers what you expect?
