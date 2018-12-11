import sys
import numpy as np
from keras.models import Sequential
from data_helper import load_data, split_data_targets
from models import NeuralNetwork, LSTMNetwork, CNNNetwork


# function that uses k-fold cross validation to evaluate a model
def cv_k_fold(model_info, k=10, verbose=0):
    # load data, targets and the splits
    data, targets, splits = load_data(k)

    # used for storing all model scores
    scores = []

    # iterate through the splits for CV
    for train_index, test_index in splits:
        # splits data and targets using indices given and returns them ready for training and testing
        (train_data, train_targets), (test_data, test_targets) = split_data_targets(data,
                                                                                    targets,
                                                                                    train_index,
                                                                                    test_index,
                                                                                    model_info.input_shape)
        # trains model and returns the score
        score = train(model_info, train_data, train_targets, test_data, test_targets, verbose)
        # store the score of this model
        scores.append(score)

        # print the accuracy metric score
        print('Fold: ' + str(len(scores)) + ', Accuracy: ' + str(round(score * 100, 3)) + '%')

    # calculate the mean score of the all the trained models
    cv_score = float(np.mean(scores) * 100)
    cv_std = float(np.std(scores) * 100)

    # print the CV accuracy score
    print('Final Accuracy:', str(round(cv_score, 3)) + '%', '(+/-', str(round(cv_std, 3)) + '%)')


# function that trains a NN model with given configuration
def train(model_info, train_data, train_targets, test_data, test_targets, verbose):
    # initialise structure of model
    model = Sequential(model_info.get_structure())

    # configure optimiser, loss and metrics
    model.compile(optimizer=model_info.optimizer, loss=model_info.loss, metrics=['accuracy'])

    # trains the model by fit the train data and targets. configure number of epochs
    model.fit(train_data, train_targets, epochs=model_info.epochs, verbose=verbose, batch_size=model_info.batch_size)

    # evaluate the trained model using test parts
    score = model.evaluate(test_data, test_targets, verbose=verbose)

    return score[1]


# main function to run when script is called
def main():
    # remove first system argument
    args = sys.argv[1:]
    # variable for number of arguments
    num_args = len(args)

    # check number of arguments passed to script is correct
    if num_args != 1 and num_args != 3:
        # helpful messages to help user
        print('Error, was expecting 1 argument or 3 arguments: <model_type> [<n_splits> <verbose>]')
        print('Found:', num_args)
        return

    # if 3 arguments are passed then check they are correct
    if num_args == 3:
        # second argument must be digit
        if not args[1].isdigit():
            print('Error, <n_splits> was expecting: k > 1')
            print('Found:', args[1])
            return
        # second argument must be greater than 1
        if int(args[1]) <= 1:
            print('Error, <n_splits> was expecting: k > 1')
            print('Found:', args[1])
            return
        # third argument must be a digit
        if not args[2].isdigit():
            print('Error, <verbose> was expecting: 0 (off) or 1 (on)')
            print('Found:', args[2])
            return
        # third argument must be a 0 or 1
        if int(args[2]) != 0 and int(args[2]) != 1:
            print('Error, <verbose> was expecting: 0 (off) or 1 (on)')
            print('Found:', args[2])
            return

    # checks if thr first argument is a valid model_type
    if args[0] == 'nn':
        if num_args == 3:
            # cross validate neural network model with args
            cv_k_fold(NeuralNetwork(), int(args[1]), int(args[2]))
        else:
            # cross validate neural network model
            cv_k_fold(NeuralNetwork())
    elif args[0] == 'lstm':
        if num_args == 3:
            # cross validate LSTM network model with args
            cv_k_fold(LSTMNetwork(), int(args[1]), int(args[2]))
        else:
            # cross validate LSTM network model
            cv_k_fold(LSTMNetwork())
    elif args[0] == 'cnn':
        if num_args == 3:
            # cross validate CNN network model with args
            cv_k_fold(CNNNetwork(), int(args[1]), int(args[2]))
        else:
            # cross validate CNN network model
            cv_k_fold(CNNNetwork())
    else:
        # first argument is not valid
        # message displays list of possible model_types
        print('Error, <model_type> was expecting: \'nn\', \'lstm\', \'cnn\'')
        print('Found: \'' + args[0] + '\'')


# main program cross validates given neural network model
if __name__ == "__main__":
    main()
