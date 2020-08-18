import argparse
import numpy as np
import pickle
import os

from knn import kNN
from vector import Vector

def train(k, data_dir, test=False):
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

    # initialize the knn model
    model = kNN(k)

    # TODO: fit the model
    model.fit(x_train, y_train)

    # evaluate the model on the evaluation data
    if test:
        x_eval, y_eval = x_test, y_test
    else:
        x_eval, y_eval = x_valid, y_valid

    num_correct = 0
    num_eval = x_eval.shape[0]
    for i in range(num_eval):
        # TODO: query the model
        # TODO: increment num_correct if correct
        query = Vector(x_eval.values[i])

        prediction = model.predict(query)
        if(prediction == y_eval.values[i]):
            num_correct += 1

    # print the evaluation results
    accuracy = num_correct/num_eval
    eval_name = 'test' if test else 'valid'
    print("Accuracy with k = {} on {} is {}".format(k, eval_name, accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int,
                       help='Hyperparameter k for kNN')
    parser.add_argument('-d', '--data-dir', type=str,
                       help='Path to data directory')
    parser.add_argument('-t', '--test', action='store_true',
                       help='whether to evaluate on the test set. '
                            'If not set, will use validation set.')
    args = parser.parse_args()

    train(args.k, args.data_dir, args.test)
