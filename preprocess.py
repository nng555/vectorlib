import argparse
import csv
import numpy as np
import os
import pickle

from vector import Vector

"""
This file will preprocess our input data into a train validation
and test split, then dump it to a set of files.
"""

def preprocess(input_file, output_path, train_frac, valid_frac):
    """
    Preprocess the input csv and separate out the features and labels.
    Split

    input_file - path to input csv `iris.data`
    output_path - path to output data folder
    train_frac - fraction of data for training split
    valid_frac - fraction of data for validation split
    """

    def process_row(row):
        """
        Process a single row and return the features and label

        row - input data row from csv

        returns:
            feat [float] - list of input features
            label [int] - output label
        """
        # row already list bc it was read with a csv.reader
        # features are the first 4 elements
        print (row)
        feat = row[:4]
        # label is the last element
        label = row[4]
        return feat, label

        #raise NotImplementedError

    # load data from csv file and process each row
    feats, labels = list(), list()
    with open(input_file, 'r') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            feat, label = process_row(row)
            feats.append(feat)
            labels.append(label)

    # TODO: shuffle the data
    # shuffle the 2 lists in unison
    feats_shuffle, labels_shuffle = sklearn.utils.shuffle(feats, shuffle)

    # TODO: split data into training, validation, and test splits
    test_frac = 1.0 - train_frac - valid_frac

    # num of examples per fraction
    numTest = test_frac*len(labels)
    numTrain = train_frac*len(labels)
    numValid = valid_frac*len(labels)

    # TODO: turn each train/valid/test feat and label list into a Vector
    x_train = Vector(feats_shuffle[:numTrain])
    y_train = Vector(labels_shuffle[:numTrain])
    x_valid = Vector(feats_shuffle[numTrain:numValid])
    y_valid = Vector(labels_shuffle[numTrain:numValid])
    x_test = Vector(feats_shuffle[numValid:numTest])
    y_test = Vector(labels_shuffle[numValid:numTest])

    data_vectors = [x_train, y_train,
                    x_valid, y_valid,
                    x_test, y_test]

    # dump each object to a pickle file
    fnames = ['x_train', 'y_train',
              'x_valid', 'y_valid',
              'x_test', 'y_test']

    for data_vector, fname in zip(data_vectors, fnames):
        with open(os.path.join(output_path, fname + '.vec'), 'wb') as of:
            pickle.dump(data_vector, of)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                       help='Input csv file for preprocessing')
    parser.add_argument('-o', '--output', type=str,
                       help='Output path for processed objects')
    parser.add_argument('-t', '--train', type=float,
                       help='Fraction of data in training split')
    parser.add_argument('-v', '--validation', type=float,
                       help='Fraction of data in validation split. '
                            'Remaining fraction is used for test split.')
    args = parser.parse_args()

    if args.train + args.validation > 1:
        raise Exception('Train and valid split cannot be more than 1.0')

    preprocess(args.input, args.output, args.train, args.validation)
