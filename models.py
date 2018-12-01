from keras.layers import Dense

# all NNs currently have the same optimizer, loss and epochs


# first configuration with:
# 1 hidden layer of 100 units
# input shape of (784,)
class NeuralNetwork:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 10

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Dense(units=100, activation='relu', input_dim=784),
            Dense(units=10, activation='softmax')
        ]
        return model_structure


# second configuration with:
# 1 hidden layer of 500 units
# input shape of (784,)
class NeuralNetwork2:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 3

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Dense(units=500, activation='relu', input_dim=784),
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

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Dense(units=100, activation='relu', input_dim=784),
            Dense(units=100, activation='relu'),
            Dense(units=10, activation='softmax')
        ]
        return model_structure
