from keras.layers import Dense, LSTM, Conv2D, ZeroPadding2D, AveragePooling2D, Flatten, MaxPooling2D

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
# 1 hidden lstm layer of 256 units
# input shape of (28, 28)
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


# CNN configuration with:
# LeNet-5 architecture 1998 was used
# input shape of (28, 28, 1) (rows, cols, channels)
class CNNNetwork:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 30
    input_shape = (28, 28, 1)
    batch_size = 64

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            ZeroPadding2D(padding=(2, 2)),
            Conv2D(6, (5, 5), activation='relu'),
            AveragePooling2D(strides=(2, 2)),
            Conv2D(16, (5, 5), activation='relu'),
            AveragePooling2D(strides=(2, 2)),
            Conv2D(120, (5, 5), activation='relu'),
            Flatten(),
            Dense(units=84, activation='relu'),
            Dense(units=10, activation='softmax')
        ]
        return model_structure


# CNN configuration with:
# LeNet-5 1998 inspired architecture was used
# input shape of (28, 28, 1) (rows, cols, channels)
class CNNNetwork2:
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    epochs = 30
    input_shape = (28, 28, 1)
    batch_size = 64

    # function that returns the model structure
    @staticmethod
    def get_structure():
        model_structure = [
            Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=CNNNetwork2.input_shape),
            Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=10, activation='softmax')
        ]
        return model_structure
