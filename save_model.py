import os
import sys
from keras.models import Sequential
from models import NeuralNetwork, LSTMNetwork, CNNNetwork
from data_helper import load_data, normalise


# trains a model with it's own config then it saves it to a file
def save(model_info, model_filename):
    # load data, targets and ignore the splits
    data, targets = load_data(k=1)

    # normalise data
    data = normalise(data, model_info.input_shape)

    # initialise structure of model
    model = Sequential(model_info.get_structure())

    # configure optimiser, loss and metrics
    model.compile(optimizer=model_info.optimizer, loss=model_info.loss, metrics=['accuracy'])

    # trains the model by fitting the data and targets. Configure number of epochs
    model.fit(data, targets, epochs=model_info.epochs, batch_size=model_info.batch_size)

    # saves the trained model and creates a HDF5 file '<model_name>.h5' at '../models/'
    model.save('../models/' + model_filename + '.h5')


# main function to run when script is called
def main():
    # get directory from path
    directory = os.path.dirname('../models/file.h5')

    # checks if directory exits
    if not os.path.exists(directory):
        # if it does not then create it
        print('Creating directory \'../models/\'')
        os.makedirs(directory)

    # check number of arguments passed to script is correct
    if len(sys.argv) != 3:
        # helpful messages to help user
        print('Error, was expecting two arguments: <model_type> <model_name>')
        print('Found:', len(sys.argv) - 1)
        return

    # checks if first argument is a valid model_type
    if sys.argv[1] == 'nn':
        # save neural network model with filename as second argument
        save(NeuralNetwork(), sys.argv[2])
        print('File successfully saved at \'../models/' + sys.argv[2] + '.h5\'')
    elif sys.argv[1] == 'lstm':
        # save lstm network model with filename as second argument
        save(LSTMNetwork(), sys.argv[2])
        print('File successfully saved at \'../models/' + sys.argv[2] + '.h5\'')
    elif sys.argv[1] == 'cnn':
        # save cnn network model with filename as second argument
        save(CNNNetwork(), sys.argv[2])
        print('File successfully saved at \'../models/' + sys.argv[2] + '.h5\'')
    else:
        # first argument is not valid
        # message displays list of possible model_types
        print('Error, <model_type> was expecting: \'nn\', \'lstm\'')
        print('Found: \'' + sys.argv[1] + '\'')


if __name__ == '__main__':
    main()
