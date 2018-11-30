from keras.layers import Dense

# all NNs currently have the same optimizer, loss, metric and epochs


# first configuration with:
# 1 hidden layer of 100 units
# input shape of (784,)
class NeuralNetwork:
    model_structure = [
        Dense(units=100, activation='relu', input_dim=784),
        Dense(units=10, activation='softmax')
    ]
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    epochs = 3


# second configuration with:
# 1 hidden layer of 500 units
# input shape of (784,)
class NeuralNetwork2:
    model_structure = [
        Dense(units=500, activation='relu', input_dim=784),
        Dense(units=10, activation='softmax')
    ]
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    epochs = 3


# third configuration with:
# 2 hidden layers of 100 units
# input shape of (784,)
class NeuralNetwork3:
    model_structure = [
        Dense(units=100, activation='relu', input_dim=784),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
    ]
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    epochs = 3
