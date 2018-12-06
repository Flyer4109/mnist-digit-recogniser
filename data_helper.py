import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical


# function used to load data and returns appropriate numpy arrays
def load_data(k=10, shuffle=True):
    # read data set from csv file and creates Pandas DataFrame
    data_set = pd.read_csv("../data/train.csv")

    # shuffles the DataFrame
    data_set = data_set.sample(frac=1).reset_index(drop=True)

    # drops labels and gets all the data as floats in numpy array in shape (42000, 784)
    data = data_set.drop(labels='label', axis=1).iloc[:].values.astype(float)

    # gets all the labels in numpy array in shape (42000,)
    targets_raw = data_set['label']

    # turns targets into binary matrices in shape (num_targets, 10)
    targets = to_categorical(targets_raw, num_classes=10)

    # if k is 1 there is no need to split
    if k > 1:
        # uses Stratified KFold and gets splits
        strat_k_fold = StratifiedKFold(n_splits=k, shuffle=shuffle)
        splits = strat_k_fold.split(data, targets_raw)

        # returns data, targets and splits
        return data, targets, splits

    # returns only data and targets
    return data, targets


# splits data and targets using train and test indices then normalises train and test data
def split_data_targets(data, targets, train_index, test_index, shape):
    # split data into train and test then normalises the data
    train_data, test_data = normalise(data[train_index], shape, data[test_index])

    # split targets into train and test
    train_targets = targets[train_index]
    test_targets = targets[test_index]

    # return all train and test sets needed for training
    return (train_data, train_targets), (test_data, test_targets)


# normalises the data by using scaling
def normalise(train, shape, test=None):
    # initialise scaler and fit TRAIN DATA first
    scaler = StandardScaler().fit(train)

    # normalise train data
    train_data = scaler.transform(train)

    # if-statement to check whether test data or shape was given to return appropriately
    if test is not None:
        # normalise test data
        test_data = scaler.transform(test)

        if shape is not None:
            # if the shape parameter is given then reshape train and test data
            return np.reshape(train_data, shape), np.reshape(test_data, shape)
        else:
            # returns normalised train and test data
            return train_data, test_data
    else:
        if shape is not None:
            # if the shape parameter is given then reshape the data
            return np.reshape(train_data, shape)
        else:
            # returns normalised train data
            return train_data
