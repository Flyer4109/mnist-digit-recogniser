import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


# function used to load data and returns appropriate numpy arrays
def load_data(k=10):
    # read data set from csv file and creates Pandas DataFrame
    data_set = pd.read_csv("../data/train.csv")

    # calculates the split location between train and test data
    cut_index = int(len(data_set.index)/k)

    # drops labels and gets all the data in numpy array in shape (42000, 784)
    data = data_set.drop(labels='label', axis=1).iloc[:].values

    # cut data into training and testing parts using cut_index
    training_data = data[:-cut_index]
    testing_data = data[-cut_index:]

    # initialise scaler and fit train data into scaler
    scaler = StandardScaler().fit(training_data)

    # transform train and test data into normalised forms
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)

    # reshape train and test data to (num_data, 784)
    training_data = np.reshape(training_data, (len(data_set.index)-cut_index, 784))
    testing_data = np.reshape(testing_data, (cut_index, 784))

    # gets all the labels in numpy array in shape (42000,)
    targets = data_set['label']

    # cut targets into training and testing parts using cut_index
    training_targets = targets[:-cut_index]
    testing_targets = targets[-cut_index:]

    # gets train and test targets in numpy array in shape (num_targets, 10)
    training_targets = to_categorical(training_targets, num_classes=10)
    testing_targets = to_categorical(testing_targets, num_classes=10)

    # returns everything needed for training and testing in 2 tuples
    return (training_data, training_targets), (testing_data, testing_targets)
