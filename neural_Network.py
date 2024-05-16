import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.activation_function = activation_function

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, output_error, learning_rate):
        self.error = output_error * ActivationFunction.sigmoid_derivative(self.output)
        self.weights -= learning_rate * np.dot(self.input.T, self.error)
        self.bias -= learning_rate * np.sum(self.error, axis=0, keepdims=True)
        return np.dot(self.error, self.weights.T)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y, output, learning_rate):
        output_error = output - y
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)
