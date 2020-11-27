import argparse
import csv
import numpy as np
import os
import pickle
import sklearn
#from sklearn.utils import shuffle

from vector import Vector

"""
This file will preprocess our input data into a train validation
and test split, then dump it to a set of files.
"""

def preprocess(dataset, output_path, train_frac, valid_frac, normalize, seed):
    """
    Preprocess the input csv and separate out the features and labels.
    Split

    input_file - path to input csv `iris.data`
    output_path - path to output data folder
    train_frac - fraction of data for training split
    valid_frac - fraction of data for validation split
    normalize - whether or not to normalize the feature vectors
    seed - random seed to use for data splits
    """

    # set the random seed
    np.random.seed(seed)

    def process_iris_row(row):
        """
        Process a single row of iris data and return the features and label

        row - input data row from csv

        returns:
            feat [float] - list of input features
            label [int] - output label
        """
        # row already list bc it was read with a csv.reader
        # features are the first 4 elements
        #print (row)
        feat = [float(val) for val in row[:4]]
        # label is the last element
        label = row[4]

        return feat, label

    def process_wine_row(row):
        """
        Process a single row of wine data and return the features and label

        row - input data row from csv

        returns:
            feat [float] - list of input features
            label [int] - output label
        """
        # TODO: Process a row of wine data. Look at the web page and the actual data
        #       to see exactly how to do this. In this case its okay to hard code in
        #       your indices.
        label = int(row[0])
        feat = [float(val) for val in row[1:]]

        return feat, label

    def process_bike_row(row):
        """
        Process a single row of bike data and return the features and label
        """
        # Process a row of bike data. Throw away the date field
        label = [int(row[1])]
        feat = [float(val) for val in row[2:-3]]

        # TODO: Add a categorical feature for which season it is. This information is
        # contained in row[-3]. Each season should get its own feature.
        if row[-3] == 'Spring':
            feat[-3] =[1,0,0,0]
        elif row[-3] == 'Summer':
            feat[-3] = [0,1,0,0]
        elif row[-3] == 'Autumn':
            feat[-3] = [0,0,1,0]
        else:
            feat[-3] = [0,0,0,1]

        # TODO: Add a binary feature for whether there is a holiday or not. This
        # information is contained in row[-2]
        if row[-2] == 'Holiday':
            feat[-2]=1
        else:
            feat[-2]=0

        # TODO: Add a binary feature for whether the bikes are working or not. This
        # information is contained in row[-1]
        if row[-1]== 'Yes':
            feat[-1] = 1
        else:
            feat[-1] = 0
        # TODO: Add an extra feature for our bias
        feat.append(1)

        return feat, label

    # load data from csv file and process each row
    feats, labels = list(), list()
    with open(dataset + '.data', 'r') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if dataset == 'iris':
                feat, label = process_iris_row(row)
            elif dataset == 'wine':
                feat, label = process_wine_row(row)
            elif dataset == 'bike':
                feat, label = process_bike_row(row)
            else:
                raise Exception("Dataset not supported!")
            feats.append(feat)
            labels.append(label)

    # shuffle the 2 lists in unison
    feats_shuffle, labels_shuffle = sklearn.utils.shuffle(feats, labels)
    feats_shuffle = Vector(feats_shuffle)
    labels_shuffle = Vector(labels_shuffle)

    if normalize:
        if dataset == 'bike':
            start = 0
            end = 9
        else:
            start = 0
            end = x_train.shape[0]
        feats_shuffle.normalize(start, end)

    # TODO: split data into training, validation, and test splits
    test_frac = 1.0 - train_frac - valid_frac

    # num of examples per fraction
    numTest = int(test_frac * len(labels))
    numTrain = int(train_frac * len(labels))
    numValid = int(valid_frac * len(labels))

    # TODO: turn each train/valid/test feat and label list into a Vector
    x_train = Vector(feats_shuffle[:numTrain])
    y_train = Vector(labels_shuffle[:numTrain])
    x_valid = Vector(feats_shuffle[numTrain:numValid + numTrain])
    y_valid = Vector(labels_shuffle[numTrain:numValid + numTrain])
    x_test = Vector(feats_shuffle[numValid + numTrain:])
    y_test = Vector(labels_shuffle[numValid + numTrain:])


    data_vectors = [x_train, y_train,
                    x_valid, y_valid,
                    x_test, y_test]

    # dump each object to a pickle file
    fnames = ['x_train', 'y_train',
              'x_valid', 'y_valid',
              'x_test', 'y_test']

    for data_vector, fname in zip(data_vectors, fnames):
        with open(os.path.join(output_path, fname + '.vec'), 'wb') as of:
            print("Dumping {} with shape {}".format(fname, data_vector.shape))
            pickle.dump(data_vector, of)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                       help='Dataset to process. Must be one of [iris, wine]')
    parser.add_argument('-o', '--output', type=str,
                       help='Output path for processed objects')
    parser.add_argument('-t', '--train', type=float,
                       help='Fraction of data in training split')
    parser.add_argument('-v', '--validation', type=float,
                       help='Fraction of data in validation split. '
                            'Remaining fraction is used for test split.')
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Whether to normalize featuers or not.')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed to use')
    args = parser.parse_args()

    if args.train + args.validation > 1:
        raise Exception('Train and valid split cannot be more than 1.0')

    preprocess(args.dataset, args.output, args.train, args.validation, args.normalize, args.seed)
