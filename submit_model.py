import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from data_helper import load_data, normalise
from models import NeuralNetwork, LSTMNetwork, CNNNetwork, CNNNetwork2


# function to create submission csv file
def submit(model_info, model_filename):
    # read test.csv and input into Pandas DataFrame
    test_set = pd.read_csv("../data/test.csv")

    # gets all the data in numpy array in shape (28000,) and convert type to float
    test_set = test_set.iloc[:].values.astype(float)

    # load training data
    data, _ = load_data(k=1)

    # normalise data to predict by using training data
    _, test_data = normalise(data, model_info.input_shape, test_set)

    # load model from .h5 file
    model = load_model('../models/' + model_filename + '.h5')

    # get predictions
    predictions = model.predict(test_data)

    # get class predictions by getting the argmax in axis-1
    class_predictions = np.argmax(predictions, axis=1)

    # create new DataFrame with ImageId (1-28000) and respective Label
    submissions = pd.DataFrame({'ImageId': list(range(1, len(class_predictions) + 1)), 'Label': class_predictions})

    # convert DataFrame into csv file while keeping header names as they are needed for submission
    submissions.to_csv('../submissions/' + model_filename + '.csv', index=False, header=True)


# main function to run when script is called
def main():
    # get directory from path
    directory = os.path.dirname('../submissions/file.csv')

    # checks if directory exits
    if not os.path.exists(directory):
        # if it does not then create it
        print('Creating directory \'../submissions/\'')
        os.makedirs(directory)

    # check number of arguments passed to script is correct
    if len(sys.argv) != 3:
        # helpful messages to help user
        print('Error, was expecting two arguments: <model_type> <model_file>')
        print('Found:', len(sys.argv) - 1)
        return

    # checks if first argument is a valid model_type
    if sys.argv[1] == 'nn':
        # create submission csv for neural network model with given .h5 file
        submit(NeuralNetwork(), sys.argv[2])
        print('File successfully saved at \'../submissions/' + sys.argv[2] + '.csv\'')
    elif sys.argv[1] == 'lstm':
        # create submission csv for lstm network model with given .h5 file
        submit(LSTMNetwork(), sys.argv[2])
        print('File successfully saved at \'../submissions/' + sys.argv[2] + '.csv\'')
    elif sys.argv[1] == 'cnn':
        # create submission csv for cnn network model with given .h5 file
        submit(CNNNetwork(), sys.argv[2])
        print('File successfully saved at \'../submissions/' + sys.argv[2] + '.csv\'')
    elif sys.argv[1] == 'cnn2':
        # create submission csv for cnn network model with given .h5 file
        submit(CNNNetwork(), sys.argv[2])
        print('File successfully saved at \'../submissions/' + sys.argv[2] + '.csv\'')
    else:
        # first argument is not valid
        # message displays list of possible model_types
        print('Error, <model_type> was expecting: \'nn\', \'lstm\', \'cnn\'')
        print('Found: \'' + sys.argv[1] + '\'')


# main program loads given model file and gets predictions in csv file
if __name__ == '__main__':
    main()
