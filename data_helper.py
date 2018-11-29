import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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

    # reshape train data to (num_train_data, 784)
    training_data = np.reshape(training_data, (len(data_set.index)-cut_index, 784))

    # gets train targets in numpy array in shape (num_train_targets)
    training_targets = data_set['label'][:-cut_index]

    # reshape test data to (num_test_data, 784)
    testing_data = np.reshape(testing_data, (cut_index, 784))

    # gets test targets in numpy array in shape (num_test_targets)
    testing_targets = data_set['label'][-cut_index:]

    # returns everything needed for training and testing in 2 tuples
    return (training_data, training_targets), (testing_data, testing_targets)
