import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical


# function used to load data and returns appropriate numpy arrays
def load_data(k=10, shuffle=True, shape=None):
    # read data set from csv file and creates Pandas DataFrame
    data_set = pd.read_csv("../data/train.csv")

    # shuffles the DataFrame
    data_set = data_set.sample(frac=1).reset_index(drop=True)

    # drops labels and gets all the data as floats in numpy array in shape (42000, 784)
    data = data_set.drop(labels='label', axis=1).iloc[:].values.astype(float)

    # reshapes the data if the shape parameter is given
    if shape is not None:
        data = np.reshape(data, (data.shape[0],) + shape)

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
def split_data_targets(data, targets, train_index, test_index):
    # split data into train and test then normalises the data
    train_data, test_data = normalise(data[train_index], data[test_index])

    # split targets into train and test
    train_targets = targets[train_index]
    test_targets = targets[test_index]

    # return all train and test sets needed for training
    return (train_data, train_targets), (test_data, test_targets)


# normalises the data by using scaling
def normalise(train, test=None):
    original_shape = train.shape
    print(original_shape)
    odd_shape = original_shape != (original_shape[0], 784)

    if odd_shape:
        train_data = np.reshape(train, (original_shape[0], 784))
        print(train_data.shape)
    else:
        train_data = train

    # initialise scaler and fit TRAIN DATA first
    scaler = StandardScaler().fit(train_data)

    train_data = scaler.transform(train_data)

    if odd_shape:
        train_data = np.reshape(train_data, original_shape)

    if test is not None:
        if odd_shape:
            test_data = np.reshape(test, (test.shape[0], 784))
        else:
            test_data = test
        test_data = scaler.transform(test_data)
        if odd_shape:
            test_data = np.reshape(test_data, test.shape)
        # normalise both train and test data and return
        return train_data, test_data
    else:
        # normalise train data and return
        return train_data
