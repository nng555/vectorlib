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
        # splitting the row into list separated by comma
        info = row.split(',')
        # features are the first 4 elements
        feat = info[:4]
        # label is the last element
        label = info[4]

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

    # do I need to write my own shuffle function or is this ok?
    random.shuffle(feats)
    random.shuffle(labels)

    # TODO: split data into training, validation, and test splits
    test_frac = 1.0 - train_frac - valid_frac

    # TODO: turn each train/valid/test feat and label list into a Vector

    # not sure if I did this correctly - I might've misunderstood what needed
    # to be done
    x_train = Vector(feats[:train_frac])
    y_train = Vector(labels[:train_frac])
    x_valid = Vector(feats[train_frac:valid_frac])
    y_valid = Vector(labels[train_frac:valid_frac])
    x_test = Vector(feats[valid_frac:test_frac])
    y_test = Vector(labels[valid_frac:test_frac])
    
    data_vectors = [x_train, y_train,
                    x_valid, y_valid,
                    x_test, y_test]

    # dump each object to a pickle file
    fnames = ['x_train', 'y_train',
              'x_valid', 'y_valid',
              'x_test', 'y_test']

    for data_vector, fname in zip(data_vectors, fnames):
        with open(os.path.join(output_path, fname), 'wb') as of:
            pickle.dump(data_vector, of)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.addArgument('-i', '--input', type=str,
                       help='Input csv file for preprocessing')
    parser.addArgument('-o', '--output', type=str,
                       help='Output path for processed objects')
    parser.addArgument('-t', '--train', type=float,
                       help='Fraction of data in training split')
    parser.addArgument('-v', '--validation', type=float,
                       help='Fraction of data in validation split. '
                            'Remaining fraction is used for test split.')
    args = parser.parse_args()

    if args.train + args.valid > 1:
        raise Exception('Train and valid split cannot be more than 1.0')

    preprocess(args.input, args.output, args.train, args.valid)
