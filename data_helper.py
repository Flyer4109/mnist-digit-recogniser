import pandas as pd
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
    targets = data_set['label']

    # uses Stratified KFold and gets splits
    strat_k_fold = StratifiedKFold(n_splits=k, shuffle=shuffle)
    splits = strat_k_fold.split(data, targets)

    # turns train and test targets into binary matrices in shape (num_targets, 10)
    targets = to_categorical(targets, num_classes=10)

    # returns data, targets and splits
    return data, targets, splits


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
def normalise(train_data, test_data=None):
    # initialise scaler and fit TRAIN DATA first
    scaler = StandardScaler().fit(train_data)

    if test_data is not None:
        # normalise both train and test data and return
        return scaler.transform(train_data), scaler.transform(test_data)
    else:
        # normalise train data and return
        return scaler.transform(train_data)
