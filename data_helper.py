import pandas as pd
import numpy as np


# function used to load data and returns appropriate numpy arrays
def load_data(k=10):
    # read dataset from csv file and creates Pandas DataFrame
    dataset = pd.read_csv("../data/train.csv")

    # calculates the split location between train and test data
    cut_index = int(len(dataset.index)/k)

    # gets train data in numpy array in shape (num_train_data, 1, 28, 28)
    training_data = np.reshape(dataset.drop(labels='label', axis=1).iloc[:-cut_index].values,
                               (len(dataset.index)-cut_index, 1, 28, 28))

    # gets train targets in numpy array in shape (num_train_targets)
    training_targets = dataset['label'][:-cut_index]

    # gets test data in numpy array in shape (num_test_data, 1, 28, 28)
    testing_data = np.reshape(dataset.drop(labels='label', axis=1).iloc[-cut_index:].values,
                              (cut_index, 1, 28, 28))

    # gets test targets in numpy array in shape (num_test_targets)
    testing_targets = dataset['label'][-cut_index:]

    # returns everything needed for training and testing in 2 tuples
    return (training_data, training_targets), (testing_data, testing_targets)
