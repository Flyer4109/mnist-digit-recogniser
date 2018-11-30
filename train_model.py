from data_helper import load_data
from models import NeuralNetwork, NeuralNetwork2, NeuralNetwork3
from keras.models import Sequential


# function that trains a NN model with given configuration
def train(model_config):
    # load data into train and test parts
    (training_data, training_targets), (testing_data, testing_targets) = load_data()

    # initialise structure of model
    model = Sequential(model_config.model_structure)

    # configure optimiser, loss and metrics
    model.compile(optimizer=model_config.optimizer, loss=model_config.loss, metrics=model_config.metrics)

    # trains the model by fit the train data and targets. configure number of epochs
    model.fit(training_data, training_targets, epochs=model_config.epochs)

    # evaluate the trained model using test parts
    score = model.evaluate(testing_data, testing_targets)

    # print the accuracy metric score
    print('Accuracy: ' + str(round(score[1]*100, 3)) + '%')


# main program trains 3 neural network models
if __name__ == "__main__":
    # trains the first NN
    print('First NN')
    train(NeuralNetwork())

    # trains the second NN
    print('Second NN')
    train(NeuralNetwork2())

    # trains the third NN
    print('Third NN')
    train(NeuralNetwork3())
