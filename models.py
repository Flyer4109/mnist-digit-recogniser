from keras.layers import Dense
from keras.layers import LSTM

# all NNs currently have the same optimizer, loss and epochs


# first configuration with:
# 1 hidden layer of 100 units
# input shape of (784,)
class NeuralNetwork:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 10
    input_shape = None
    batch_size = 64

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Dense(units=100, activation='relu', input_dim=784),
            Dense(units=10, activation='softmax')
        ]
        return model_structure


# LSTM configuration with:
# 1 hidden layer of 32 units
# input shape of (784, 1)
class LSTMNetwork:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 15
    input_shape = (28, 28)
    batch_size = 64

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            LSTM(256, input_shape=LSTMNetwork.input_shape),
            Dense(units=10, activation='softmax')
        ]
        return model_structure


# third configuration with:
# 2 hidden layers of 100 units
# input shape of (784,)
class NeuralNetwork3:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 3
    input_shape = None
    batch_size = 64

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Dense(units=100, activation='relu', input_dim=784),
            Dense(units=100, activation='relu'),
            Dense(units=10, activation='softmax')
        ]
        return model_structure
