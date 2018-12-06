import numpy as np
from keras.models import Sequential
from data_helper import load_data, split_data_targets
from models import NeuralNetwork, NeuralNetwork2, NeuralNetwork3


# function that uses 10-fold cross validation to evaluate a model
def cv_10_fold(model_info, verbose=0):
    # load data, targets and the splits
    data, targets, splits = load_data()

    # used for storing all model scores
    scores = []

    # iterate through the splits for CV
    for train_index, test_index in splits:
        # splits data and targets using indices given and returns them ready for training and testing
        (train_data, train_targets), (test_data, test_targets) = split_data_targets(data,
                                                                                    targets,
                                                                                    train_index,
                                                                                    test_index)
        # trains model and returns the score
        score = train(model_info, train_data, train_targets, test_data, test_targets, verbose)
        # store the score of this model
        scores.append(score)

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
    model.fit(train_data, train_targets, epochs=model_info.epochs, verbose=verbose)

    # evaluate the trained model using test parts
    score = model.evaluate(test_data, test_targets, verbose=verbose)

    if verbose == 1:
        # print the accuracy metric score
        print('Accuracy: ' + str(round(score[1]*100, 3)) + '%')

    return score[1]


# main program trains 3 neural network models
if __name__ == "__main__":
    # trains the first NN
    print('First NN')
    cv_10_fold(NeuralNetwork())
