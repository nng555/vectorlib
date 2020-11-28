import argparse
import numpy as np
import pickle
import os
import sklearn

from knn import kNN
from vector import Vector
from regression import Regression

def train(lr, batch_size, num_epochs, data_dir, test=False):
    """
    Train a kNN model and evaluate it. Reports accuracy.
    Training is usually done in the same format:
        1) load the data
        2) create the model
        3) train (or fit) the model to the training data
        4) evaluate the trained model on the evaluation data
    Finally, we report whatever relevant metrics we are interested in.

    k - hyperparameter k for kNN
    data_dir - directory containing training and evaluation data
    test - false if validating, true if testing
    """
    # load the data
    fnames = ['x_train', 'y_train',
              'x_valid', 'y_valid',
              'x_test', 'y_test']
    data_vectors = []
    for fname in fnames:
        with open(os.path.join(data_dir, fname + '.vec'), 'rb') as in_file:
            data = pickle.load(in_file)
            data_vectors.append(data)

    [x_train, y_train,
     x_valid, y_valid,
     x_test, y_test] = data_vectors

    # TODO: initialize the regression model
    num_features = x_train.shape[1]
    num_outputs = y_train.shape[1]

    model = Regression(num_features, num_outputs, lr)

    # TODO: fit the model
    batch_num = int(x_train.shape[0]/batch_size) + 1
    for epoch in range(num_epochs):
        print("*******EPOCH {}*******".format(str(epoch)))

        # shuffle the data
        x_train, y_train = sklearn.utils.shuffle(x_train.values, y_train.values)
        x_train = Vector(x_train)
        y_train = Vector(y_train)

        epoch_loss = 0
        for batch in range(batch_num):
            x_batch = Vector(x_train[batch * batch_size:(batch+1) * batch_size])
            y_batch = Vector(y_train[batch * batch_size:(batch+1) * batch_size])

            # TODO: make predictions on the batch
            test_pred = model.forward(x_batch)

            # TODO: get your model loss and gradient
            test_loss, test_gradient = model.get_loss(y_batch,test_pred)

            # TODO: run backpropagation
            model.backward(x_batch,test_gradient)

        # TODO: make predictions on the validation data
        valid_pred = model.forward(x_valid)
        # TODO: calculate the loss on our validation data
        valid_loss, valid_grad = model.get_loss(y_valid,valid_pred)

        print("Validation loss: {}".format(str(valid_loss)))

    print(model.weights.transform.values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float,
                       help='learning rate for backpropagation')
    parser.add_argument('-b', type=int,
                       help='batch size')
    parser.add_argument('-n', type=int,
                       help='number of epochs')
    parser.add_argument('-d', '--data-dir', type=str,
                       help='Path to data directory')
    parser.add_argument('-t', '--test', action='store_true',
                       help='whether to evaluate on the test set. '
                            'If not set, will use validation set.')
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'y_train.vec'), 'rb') as in_file:
        len = pickle.load(in_file)

    train(args.l, args.b, args.n, args.data_dir, args.test)
