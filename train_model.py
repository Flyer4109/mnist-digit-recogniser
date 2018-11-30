from data_helper import load_data, split_data_targets
from models import NeuralNetwork, NeuralNetwork2, NeuralNetwork3
from keras.models import Sequential


# function that uses 10-fold cross validation to evaluate a model
def cv_10_fold(model_config):
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
        score = train(model_config, train_data, train_targets, test_data, test_targets)
        # store the score of this model
        scores.append(score)

    # calculate the mean score of the all the trained models
    cv_score = sum(scores) / len(scores)

    # print the CV accuracy score
    print('Final Accuracy:', str(round(cv_score*100, 3)) + '%')


# function that trains a NN model with given configuration
def train(model_config, training_data, training_targets, testing_data, testing_targets):
    # initialise structure of model
    model = Sequential(model_config.get_structure())

    # configure optimiser, loss and metrics
    model.compile(optimizer=model_config.optimizer, loss=model_config.loss, metrics=model_config.metrics)

    # trains the model by fit the train data and targets. configure number of epochs
    model.fit(training_data, training_targets, epochs=model_config.epochs, verbose=0)

    # evaluate the trained model using test parts
    score = model.evaluate(testing_data, testing_targets, verbose=0)

    # print the accuracy metric score
    print('Accuracy: ' + str(round(score[1]*100, 3)) + '%')

    return score[1]


# main program trains 3 neural network models
if __name__ == "__main__":
    # trains the first NN
    print('First NN')
    cv_10_fold(NeuralNetwork())
